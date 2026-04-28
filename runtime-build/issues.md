# runtime/build.py and runtime/_allocation.py Free-Threading Issues

Initial (shallow) pass. Each triage-note entry is a candidate intended for a
follow-up deep dive — severity/tier assignments are preliminary.

Files in scope:

- `runtime/build.py` — dynamic C-module compilation used to build the native
  launcher stub and backend utility modules (`cuda_utils`, `hip_utils`,
  gsan allocator, etc.). Exposes `compile_so_from_{src,file}` /
  `compile_module_from_{src,file}`. Uses `@functools.lru_cache` on
  `_find_compiler` and `platform_key`.
- `runtime/_allocation.py` — GPU memory allocator interface. `ContextVar`-based
  primary allocator (`_allocator`) and a non-ContextVar singleton wrapper
  (`_profile_allocator` / `_AllocatorWrapper`) used by the native launch path.

## Issues

| # | Severity (prelim) | Component | Tier | Issue |
|---|-------------------|-----------|------|-------|
| 1 | Significant | `_profile_allocator` | 3 | [`has_profile_allocator()` / `_profile_allocator.get()` TOCTOU in launch hot path](profile-allocator-toctou.md) |
| 2 | Minor | `_profile_allocator` | 3 | [`_AllocatorWrapper._allocator` unsynchronized read/write (plain attribute, not `ContextVar`)](profile-allocator-wrapper-race.md) |
| 3 | Minor | `_compile_so` | 2 | [Duplicate compile / `cache.put` write race for the same cache key](compile-so-duplicate-compile.md) |
| 4 | Minor | `_compile_so` | 3 | [Non-atomic multi-read of `knobs.build.impl` / `knobs.build.backend_dirs` / `knobs.build.cc`/`cxx` during `_build`](build-knobs-multi-read.md) |
| 5 | Minor | `_find_compiler` | 1 | [`@functools.lru_cache` on `_find_compiler` freezes first-seen `CC` / `CXX` / `PATH` resolution](find-compiler-lru-cache.md) |
| 6 | Minor | `_load_module_from_path` | 2 | [Concurrent `exec_module` of the same `.so` cache file — extension-module init race under free-threading](load-module-exec-race.md) |

Nothing SEVERE surfaced on the shallow pass. All six are candidates for a
closer look; the notes below flag the specific questions a deep dive should
answer.

## Triage notes

### 1. `has_profile_allocator()` / `_profile_allocator.get()` TOCTOU

- **Code (`_allocation.py:56-68`):**
  ```python
  _profile_allocator = _AllocatorWrapper(_NULL_ALLOCATOR)

  def set_profile_allocator(allocator):
      _profile_allocator.set(allocator if allocator is not None else _NULL_ALLOCATOR)

  def has_profile_allocator():
      return not isinstance(_profile_allocator.get(), NullAllocator)
  ```
- **Readers on hot path (nvidia/backend/driver.py:317-319, amd/backend/driver.py:363-365):**
  ```python
  if _allocation.has_profile_allocator():
      profile_scratch = allocate_scratch(..., _allocation._profile_allocator)
  ```
  Inside `allocate_scratch` (nvidia/backend/driver.py:301-307):
  `alloc_fn = allocator.get(); return alloc_fn(alloc_size, align, stream)`.
- **Writer:** `proton/hooks/instrumentation.py:194, 217` —
  `set_profile_allocator(self.allocator)` / `set_profile_allocator(NullAllocator())`.
- **Concern:** `has_profile_allocator()` does one `.get()`; then
  `allocate_scratch` does a second `.get()` on the same wrapper. Between the
  two, a Proton instrumentation hook on another thread can call
  `set_profile_allocator(None)`, swapping `_allocator` back to
  `_NULL_ALLOCATOR`. Result: `alloc_fn(alloc_size, ...)` is `NullAllocator(...)`
  which raises `RuntimeError("Kernel requires a runtime memory allocation...")`
  even though the caller explicitly gated on `has_profile_allocator()`.
  Tier 3 (configuration mutation during launch) — Proton start/stop hooks
  are the intended mutator and can fire from any thread.
- **Deep-dive questions:**
  - Is the write actually expected from other threads, or does Proton guarantee
    main-thread-only activation? If single-threaded by design, drop severity.
  - Should the fix hoist a single `.get()` snapshot into a local
    (`alloc = _profile_allocator.get(); if not isinstance(alloc, NullAllocator): ...`)
    or convert `_profile_allocator` to a `ContextVar` matching `_allocator`?
    The latter changes user-visible semantics (context-local vs. process-global).
- **Suggested fix direction:** single `.get()` snapshot captured in a local
  before the launch path, or migrate `_profile_allocator` to a `ContextVar`
  (consistent with `_allocator`).

### 2. `_AllocatorWrapper._allocator` unsynchronized read/write

- **Code (`_allocation.py:37-53`):**
  ```python
  class _AllocatorWrapper:
      def __init__(self, allocator): self._allocator = allocator
      def get(self): return self._allocator
      def set(self, allocator): self._allocator = allocator
      def __call__(self, size, alignment, stream):
          return self._allocator(size, alignment, stream)
  ```
- **Concern:** `_AllocatorWrapper` implements a `ContextVar`-shaped surface
  but is *not* context-local. `set()` and `get()` just write/read a plain
  instance attribute, which under free-threading is a bare `__dict__` op. No
  memory barrier, no `ContextVar` semantics. A launch path that captures the
  wrapper (not the inner allocator) and later reads `.get()` may observe a
  different allocator than the one in effect when the caller decided to
  proceed (same root as #1).
- **Consequence beyond #1:** the wrapper's `__call__` does `self._allocator(...)`
  which loads the attribute again — so even a single `wrapper(size, align, stream)`
  call reads the attribute at the moment of invocation, not at the moment the
  wrapper was chosen. A mid-launch swap corrupts the allocator identity.
- **Deep-dive questions:**
  - Was the `_AllocatorWrapper` `.get/.set` surface deliberately non-contextvar
    (e.g., so Proton's instrumentation can affect the whole process globally)?
    The docstring says "used in same way as allocator" — but `allocator` is a
    `ContextVar`, so this is a real semantic divergence.
  - Is the intended fix to unify with `ContextVar`, or to add a lock?
- **Suggested fix direction:** same as #1 — snapshot at the call site. If
  the "process-global, not context-local" behavior is deliberate, document
  it and add a lock or atomic swap.

### 3. Duplicate compile / `cache.put` write race

- **Code (`build.py:132-154`):**
  ```python
  def _compile_so(src, src_path, name, ..., load_module, language):
      cache = _get_cache_manager(src, config=config)
      cache_path = cache.get_file(f"{name}{suffix}")
      if cache_path is not None:
          if not load_module: return cache_path
          try: return _load_module_from_path(name, cache_path)
          except (RuntimeError, ImportError):
              log.warning(...)

      with tempfile.TemporaryDirectory() as tmpdir:
          so = _build(name, src_path, tmpdir, ...)
          with open(so, "rb") as f:
              cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)
      return _load_module_from_path(name, cache_path) if load_module else cache_path
  ```
- **Concern:** Two threads (e.g., two `CudaUtils()` first-use paths in
  different subprocesses... or two backend driver inits) reach `_compile_so`
  with the same cache key and both see `cache.get_file(...) is None`. Both
  run `_build(...)` — an `O(seconds)` `subprocess.check_call(cc_cmd, ...)` —
  and both call `cache.put(...)`. The file write is atomic-replace so the
  on-disk state is consistent (last writer wins; bytes are deterministic
  given the key), but the compile work is duplicated and two threads are
  fighting for the same `tempfile.TemporaryDirectory()` scope (each has its
  own, so OK) and the same `cc_cmd` target `so` path (inside different
  tmpdirs, so OK).
- **Likely classification:** benign duplicate computation. Tier 2.
- **Deep-dive questions:**
  - Are concurrent first-use calls actually realistic? The main callers
    (`CudaUtils` / `HIPUtils`) are singleton-per-process, and the singleton
    `__new__` itself has a TOCTOU (see `third_party/nvidia/backend/driver.py:95-98`)
    that is not part of this component. Note: if both `CudaUtils()` instances
    race, both `__init__`s overwrite module-level globals (`PyCUtensorMap`, etc.)
    — that's a caller bug, not a build.py bug, but worth cross-referencing.
  - Does `_build`'s `subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)`
    interact badly with concurrent Python threads (e.g., signal handling under
    free-threading)?
- **Suggested fix direction:** none required for build.py itself. A
  coalescing lock keyed by cache key would eliminate duplicate compiles at
  the cost of holding a lock across seconds of compilation.

### 4. Non-atomic multi-read of `knobs.build.*` during `_build`

- **Code (`build.py:62-83`):**
  ```python
  if impl := knobs.build.impl:
      return impl(name, src, srcdir, library_dirs, include_dirs, libraries)
  ...
  custom_backend_dirs = knobs.build.backend_dirs   # reads cudacrt_path + cudart_path
  include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
  cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so]
  ...
  ```
  And via `_find_compiler` (line 22-47): `os.environ.get("CC")`,
  `shutil.which("clang")`, `shutil.which("gcc")`.
- **Concern:** Parallels the `cache.py` #2 triage note. Each individual
  knob read is atomic, but the sequence (`impl`, then `backend_dirs`, then
  `cc`/`cxx` inside the lru_cache) is not. Tier 3 mutation during a compile
  could yield a mix of old and new config.
- **Follow-on:** `knobs.build.backend_dirs` is itself a `@property` that
  reads `self.cudacrt_path` *and* `self.cudart_path` in a single expression
  (knobs.py:329-330). That single property evaluation is itself a two-read
  sequence within the non-atomic multi-read described above. Both reads
  inside the property go through the descriptor protocol, so each is atomic
  individually.
- **Deep-dive questions:**
  - What real workflow mutates `knobs.build.impl` at runtime? If this is
    "set once before any compile", it's not worth reporting (write-once after
    import).
  - Does `knobs.build.cc` / `cxx` get a deep-dive under the separate `knobs`
    component? If so, defer.
- **Suggested fix direction:** snapshot the knobs into locals at the top of
  `_build`, same fix pattern as in the `cache.py` triage.

### 5. `@functools.lru_cache` on `_find_compiler` freezes first-seen env

- **Code (`build.py:21-47`):**
  ```python
  @functools.lru_cache()
  def _find_compiler(language: str) -> str:
      if language == "c":
          cc = os.environ.get("CC")
          if cc is not None: return cc
          ...
  ```
- **Concern:** Two separate issues:
  1. `lru_cache` stores the first thread's answer. Any later mutation of `CC`
     / `CXX` / `PATH` is invisible for the rest of the process. This is
     GIL-era behavior too, not a free-threading regression.
  2. Under free-threading, `lru_cache` is internally synchronized (PEP 703),
     but duplicate cold-start invocations can occur before the first result
     is stored — each thread runs `shutil.which("clang")` / `shutil.which("gcc")`.
     `shutil.which` walks `PATH` and performs `os.access` syscalls; duplicating
     is benign.
- **Likely classification:** Minor / already-GIL-era behavior. Tier 1.
- **Deep-dive questions:**
  - Does `shutil.which` itself touch any shared mutable state? (Believed
    purely read-only.)
  - Is the cold-start duplicate work worth a double-checked lock? Probably not.
- **Suggested fix direction:** none. Keep the `lru_cache` as-is.

### 6. Concurrent `exec_module` of the same `.so` cache file

- **Code (`build.py:108-114`):**
  ```python
  def _load_module_from_path(name: str, path: str) -> ModuleType:
      spec = importlib.util.spec_from_file_location(name, path)
      if not spec or not spec.loader: raise RuntimeError(...)
      mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(mod)
      return mod
  ```
- **Concern:** When two threads reach `_compile_so` for the same cache key
  concurrently and both hit the cache path, both call `_load_module_from_path`,
  which runs `spec.loader.exec_module(mod)` against the same `.so` file.
  For C extension modules, the first `dlopen` triggers `PyInit_<name>`;
  subsequent loads on the same path share the same dynamic-linker handle
  but CPython's extension loader machinery creates a fresh module object
  each call. Under the GIL this serialized through the import lock; under
  free-threading, the relevant protections live inside `importlib._bootstrap_external`
  and `ExtensionFileLoader`.
- **Deep-dive questions:**
  - Is `ExtensionFileLoader.exec_module` safe for concurrent invocation on
    the same `.so` path under free-threading? CPython 3.14t has had several
    fixes in this area; confirm the current status.
  - Module init functions (`PyInit_*`) in the emitted launcher / `cuda_utils`
    / `hip_utils` set up module-level state (type objects, static tables).
    If `PyInit_*` runs twice concurrently in the same process, does the
    module's own init code expect single-threaded execution? This is a
    third-party (nvidia/amd `driver.c`) concern — cross-reference when
    those components are audited.
  - `_load_module_from_path` does *not* register the module in `sys.modules`,
    so each caller gets a fresh `ModuleType`. Is that intentional?
- **Suggested fix direction:** likely none at the build.py layer — push the
  deep-dive into the emitted-launcher audit (the `make_launcher.py`
  component and the backend `driver.c` files). If a bug is found, the fix
  might be to cache the `ModuleType` in `build.py` keyed by cache path.

## Not worth reporting (initial pass)

- **`_allocator: ContextVar[Allocator]`** (`_allocation.py:26`). `ContextVar`
  is thread-safe and context-local by design. Matches CLAUDE.md's
  `ContextVar`-based state carve-out.
- **`NullAllocator` / `_NULL_ALLOCATOR`** module-level singleton — stateless,
  immutable.
- **`@functools.lru_cache` on `platform_key()`** (`build.py:94`). Pure
  function over `platform.machine()` / `system()` / `architecture()`. Same
  internal-sync story as `_find_compiler` but with no env dependency.
- **`_language_from_filename`, `_get_file_extension`, `_library_flag`** —
  pure functions.
- **`tempfile.TemporaryDirectory()` in `_compile_so` / `_compile_so_from_src`**
  — each call creates a unique directory with a random name; no sharing.
- **`subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)`** — each
  call uses an independent process and independent pipes. Compiler
  concurrency is the OS's job.
- **Module-level imports of `knobs`, `get_cache_manager`** — write-once at
  import time.
- **`set_allocator(allocator)`** (`_allocation.py:29-34`). Writes to a
  `ContextVar`; `ContextVar.set` is thread-safe.

## Caller-side concerns noted but out of scope

- **`CudaUtils.__new__` singleton TOCTOU** (`third_party/nvidia/backend/driver.py:95-98`):
  ```python
  def __new__(cls):
      if not hasattr(cls, "instance"):
          cls.instance = super().__new__(cls)
      return cls.instance
  ```
  Two concurrent first-use calls can both pass the `hasattr` check, both
  construct, and both run `__init__`, which overwrites module-level globals
  `PyCUtensorMap`, `PyKernelArg`, `ARG_CONSTEXPR`, `ARG_KERNEL`, `ARG_TUPLE`
  (driver.py:108-117). Belongs in the `nvidia-driver` / eventual
  `amd-driver` component audit, not here. Flagged for cross-reference.
- **`compile_module_from_file` result is used to repopulate module globals**
  in backend drivers. The build step itself is fine; the post-build
  assignments are the caller's concurrency concern.
