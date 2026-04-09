# compiler/ Free-Threading Issues

Issues in `python/triton/compiler/` affecting free-threaded Python 3.14t.

Files in scope (Pass 1/Pass 4 per `AUDIT_PLAN.md`):

- `compiler/compiler.py` — `compile()`, `CompiledKernel`, `ASTSource`, `IRSource`, `AsmDict`, `LazyDict`, `max_shared_mem`
- `compiler/code_generator.py` — front-end lowering path (audit only for
  module-level shared state, module caches, or mutable globals)
- `compiler/make_launcher.py` — currently empty
- `compiler/errors.py` — exception classes only
- `compiler/__init__.py` — re-exports only

## Issues

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| 1 | SEVERE | CompiledKernel | [`CompiledKernel._init_handles` lazy handle initialization race](compiled-kernel-init-handles-race.md) |
| 2 | Significant | compile() | [`knobs.runtime.add_stages_inspection_hook` TOCTOU in `compile()`](add-stages-inspection-hook-toctou.md) |
| 3 | Significant | HookChain | [`kernel_load_start_hook` / `kernel_load_end_hook` — `HookChain` iteration race in `_init_handles`](kernel-load-hook-chain-race.md) |
| 4 | Significant | code_generator | [`CodeGenerator.__init__` iterates live `fn.__globals__` via `get_capture_scope`](code-generator-gscope-iteration-race.md) |

## Triage notes

Candidates surfaced during an initial pass over the compiler package. Issues
that look like real races on reusable wrapper objects or the hot
compile/launch path are listed first, followed by things we looked at and
decided are not worth reporting.

### 1. `CompiledKernel._init_handles` — lazy handle initialization race

Written up in
[compiled-kernel-init-handles-race.md](compiled-kernel-init-handles-race.md).

Summary: `CompiledKernel` instances live in
`JITFunction.device_caches[device].kernel_cache` (`runtime/jit.py:877`) and
are shared across threads under Tier 2. `_init_handles` publishes
`self._run = launcher_cls(...)` at `compiler.py:455` **before**
`driver.active.utils.load_binary(...)` populates `self.module` /
`self.function` at `compiler.py:468`. The `run` property
(`compiler.py:476-480`) guards on `self._run is None`, not
`self.module is None`, so a second thread arriving in the window between
lines 455 and 468 skips initialization, receives the launcher, then reads
`self.function` (still `None`) at `runtime/jit.py:761` and passes it into
the driver launch call. Additional hazards: guard TOCTOU at line 440
causing duplicate `load_binary` and duplicate `kernel_load_*_hook` calls,
and a torn tuple-unpack at line 468 that can publish `self.module` before
`self.function`. Tier 2 SEVERE.

### 2. `knobs.runtime.add_stages_inspection_hook` TOCTOU in `compile()`

Written up in
[add-stages-inspection-hook-toctou.md](add-stages-inspection-hook-toctou.md).

Summary: `compiler.py:247-249` reads
`knobs.runtime.add_stages_inspection_hook` twice — once for the `is
not None` check and once to invoke it — so a concurrent clear raises
`TypeError: 'NoneType' object is not callable` on the compile hot
path, and a concurrent swap folds the *new* hook's
`inspect_stages_key` into `key` even though the "installed" hook at
entry was the old one. Because `key` feeds the SHA-256 `hash` at
line 250 which is the filesystem cache directory key
(`get_cache_manager(hash)` at 251), the swap case produces orphaned
on-disk cache entries keyed under a hook that was not logically
current — a persistent cross-process cache-behavior consequence
rather than just in-memory staleness. Same shape as the jit.py
writeup ([`jit/add-stages-inspection-hook-toctou.md`](../jit/add-stages-inspection-hook-toctou.md))
but independent: `compile()` is reachable from `_do_compile`,
`warmup`, and external callers of `triton.compiler.compile`, so
fixing only the jit.py site leaves this one racey. Tier 3
Significant. Fix: single load into a local.

### 3. `knobs.runtime.kernel_load_start_hook` / `kernel_load_end_hook` — `HookChain` iteration race

Written up in
[kernel-load-hook-chain-race.md](kernel-load-hook-chain-race.md).

Summary: The initial triage framed this as the same slot-reassignment
TOCTOU as #2, but closer reading shows the real bug is different.
`kernel_load_start_hook` and `kernel_load_end_hook` are declared as
`HookChain[InitHandleHook]` instances at `knobs.py:470-471`, not
`Optional[callable]`, and are **never reassigned** anywhere in the
tree — all callers use `.add()` / `.remove()` (Proton
`hook.py:101,126`, `test_launch.py`). The `is not None` check at
`compiler.py:465,473` is effectively dead and the slot-TOCTOU is
moot in practice.

The actual race is **inside `HookChain.__call__`** (`knobs.py:422-424`):
it iterates `self.calls` — a plain `list` — while concurrent
`.add` / `.remove` mutate that same list. Same class of bug as
jit.py #7 (`pre_run_hooks`), same fix pattern, but worse in scope:
`HookChain` here is process-global, so any concurrent
`_init_handles` on *any* `CompiledKernel` collides with any
concurrent profiler start/stop. Consequences: silently skipping or
double-invoking a kernel-load hook, wrong/missing profiler
attribution. Compounds with issue #1: when two threads race
`_init_handles`, each duplicate hook invocation is independently
vulnerable to the iteration race. The same bug also affects
`launch_enter_hook` / `launch_exit_hook` on the launch path
(`compiler.py:502`, `runtime/jit.py:762`), which are also
`HookChain`s.

Separately, `HookChain.add` / `.remove` have check-then-act
compounds (dedup check followed by append; membership check
followed by remove) that allow duplicate adds and a `ValueError`
on racing removes — minor relative to the iteration race.

Tier 3 Significant. Fix: a single ~25-line copy-on-write edit in
`HookChain` (one lock for writers, snapshot-load in `__call__`)
repairs all four hook slots at once.

### 4. `max_shared_mem` — `functools.lru_cache` on the launch path

- **File/lines:** `compiler.py:133-135`.
  ```python
  @functools.lru_cache()
  def max_shared_mem(device):
      return driver.active.utils.get_device_properties(device)["max_shared_mem"]
  ```
- **Call site:** `CompiledKernel._init_handles` at line 457, i.e. the Tier 2
  hot path where two threads can concurrently first-launch the same
  `CompiledKernel` (see issue #1).
- **Deeper analysis:**
  1. **Cache container safety.** On free-threaded CPython 3.13+, the
     `_functools` C implementation of `lru_cache` guards the underlying
     cache dict and linked list with a per-wrapper critical section.
     Concurrent hits and misses will not corrupt the cache. Confirmed as
     the expected guarantee by the
     [py-free-threading porting guide](https://py-free-threading.github.io/porting/),
     which is referenced in `PYTHON_THREADSAFETY.md`.
  2. **Single-evaluation is not guaranteed.** `lru_cache` does **not**
     promise "wrapped function called at most once per key." Two threads
     that both miss on the same `device` will both execute the body, then
     both store results (last writer wins). Each thread returns its own
     computed value. This is a general `lru_cache` property, not a
     free-threading regression.
  3. **Duplicate body execution is benign here.** The body is
     `driver.active.utils.get_device_properties(device)["max_shared_mem"]`,
     which is a pure hardware query returning an `int` that is fixed for
     the life of the process (maximum shared memory per block is a
     compile-time property of the GPU). Two calls return the same value,
     so duplicate evaluation cannot corrupt the result `_init_handles`
     observes. The `device` key space is bounded by physical device count,
     so unbounded growth is not a concern either.
  4. **Transitive singletons are out of scope for this item.** The body
     does touch two racy lazy singletons: `driver.active` (lazy
     `_active is None` init at `runtime/driver.py:37-40`) and
     `CudaUtils.__new__` / `__init__` (racy `hasattr` singleton at
     `third_party/nvidia/backend/driver.py:60-90`, which also re-runs
     `compile_module_from_src` and re-binds module-level globals on every
     `__init__`). Both are real races, but they are reachable from many
     other call sites — every `driver.active.*` access in `_init_handles`
     and `compile()` hits the same pattern — and they belong to the
     `runtime/driver.py` and backend audit passes, not to
     `compiler.py:133`. `max_shared_mem` is not the site to report them.
  5. **Staleness.** `max_shared_mem` is a hardware constant. No scenario
     in Triton's runtime invalidates it, so the standard "cached value
     becomes wrong after a config change" concern does not apply.
- **Verdict: not a bug on its own.** No issue file created. The `lru_cache`
  behavior is safe at the container level, and the duplicate-evaluation
  window only produces one extra idempotent driver query. The underlying
  lazy-driver and `CudaUtils` singleton races are logged against
  `runtime/driver.py` and the NVIDIA backend, where they cover all call
  sites uniformly.

### 5. `AsmDict.__missing__` — lazy `sass` population

- **File/lines:** `compiler.py:390-400`. `CompiledKernel.asm` is built in
  `__init__` at line 426 and is the single instance held by a
  `CompiledKernel`.
  ```python
  class AsmDict(dict):
      def __missing__(self, key):
          if key == "sass":
              value = get_sass(self["cubin"])
          else:
              raise KeyError("Unknown key: '%s'" % key)
          self[key] = value
          return value
  ```
- **Sharing model:** `CompiledKernel` instances live in
  `JITFunction.device_caches[device].kernel_cache` and are reused across
  threads (Tier 2), so the `asm` dict is shared state. `'cubin'` is set
  once in `__init__` and never mutated afterwards; only `'sass'` is added
  lazily on first access via `kernel.asm['sass']`.
- **Deeper analysis:**
  1. **`self["cubin"]` read is safe.** It is a plain dict `__getitem__`
     on a key that was written once during `__init__` and never removed.
     Atomic under free-threading.
  2. **Concurrent `__missing__` entry is expected and safe.** Under
     free-threading, two threads that both look up the missing `'sass'`
     key can both enter `__missing__`; the dict's internal critical
     section does not extend across the Python call-out to `__missing__`.
     Both threads call `get_sass(cubin)` and both end with
     `self[key] = value`. The two setitems are each atomic and write the
     same semantic value — `get_sass` is a pure function of its cubin
     argument — so the final cached value is correct regardless of which
     store lands last. Both callers get a valid sass string from their
     own call.
  3. **`get_sass` is itself `@functools.lru_cache()`**
     (`tools/disasm.py:66-75`), keyed by `(cubin_asm, fun)` over
     immutable `bytes`. Same free-threading analysis as item #4: the
     cache dict is container-safe, single-evaluation is not guaranteed,
     but duplicate evaluation is idempotent. The wrapped body spawns
     `cuobjdump` in a subprocess via `tempfile.mkstemp` + `subprocess.
     check_output` + `os.remove`; `mkstemp` hands out unique paths per
     call, and the subprocesses do not share state. Two concurrent
     duplicate evaluations cost an extra `cuobjdump` run — expensive but
     benign.
  4. **No TOCTOU on a `'sass' in self` check.** The code does not do
     `if 'sass' not in self: self['sass'] = ...`; it relies on
     `__missing__`, which is only called on a true miss. So there is no
     check-then-act compound in user code here.
  5. **No hot-path impact.** `asm['sass']` is only read by users
     inspecting the disassembly (testing, tooling, proton). It does not
     sit on the compile or launch hot path, so even if duplicate
     `cuobjdump` execution were considered expensive it would not
     compound with compile/launch concurrency.
- **Verdict: not a bug.** The existing triage conclusion holds. No issue
  file created. The container is safe, the lazy initializer is a pure
  function of an immutable input, the final stored value is always
  correct, and the only observable cost of the race is at most one
  duplicate `cuobjdump` invocation per `CompiledKernel` lifetime.

### 6. `compile()` filesystem cache coordination

- **File/lines:** `compiler.py:244-356`. `fn_cache_manager.get_group`
  (line 265), `fn_cache_manager.put` (lines 314/317/339/354),
  `fn_cache_manager.put_group` (line 356).
- **Deeper analysis:**
  1. **Per-call state is all local.** `fn_cache_manager` is a fresh
     `FileCacheManager` built by `get_cache_manager(hash)` each call
     (no in-memory cache of managers — see `runtime/cache.py:254-256`).
     `metadata_group`, `metadata`, `stages`, `context`, `codegen_fns`,
     `module_map`, and `module` are all call-local. There is no shared
     Python object coordinating the on-disk cache within `compile()`.
  2. **`FileCacheManager.put` write pattern**
     (`runtime/cache.py:102-126`). Each `put`:
     - creates a per-call `tmp.pid_{pid}_{uuid4}/` directory,
     - writes the payload into that temp dir,
     - `os.replace(temp_path, filepath)` — atomic on POSIX,
     - `os.removedirs(temp_dir)` — only walks up while empty, so it
       stops at the first sibling and cannot delete `cache_dir` (the
       just-replaced target file is always present as a sibling).
     Two threads writing the *same* logical filename collide only at
     the final `os.replace`; each replace is atomic and the loser is
     simply overwritten. Two threads writing *different* logical
     filenames never collide — temp dirs are uuid-unique, and the
     final targets differ. No Python or container state is shared.
  3. **The `self.lock_path` attribute is a red herring.** It is
     computed in `FileCacheManager.__init__` (line 44/54) and asserted
     non-None in `put` (line 108), but it is **never** used as a lock —
     there is no `fcntl.flock`, no file creation at that path, no
     context manager. It is effectively dead except as an assertion.
     So multi-process and multi-thread coordination relies entirely
     on `os.replace` atomicity.
  4. **Group-file commit semantics.** `put_group` (lines 95-100) writes
     `__grp__{filename}` last, after all stage files are in place.
     `get_group` (line 73) uses `has_file(grp_filename)` as its
     presence check, so a reader only observes a "cache hit" after the
     group file commit. Partial state is invisible to readers that go
     through `get_group`.
  5. **The real semi-race: non-deterministic duplicate compile.** Two
     threads in the same process (or two processes) with the same
     `hash` both miss at line 265, both run the full compile pipeline,
     and both write stage files to the same on-disk paths. Because
     `os.replace` is atomic per file, each individual on-disk file is
     always self-consistent. But the **set** of stage files on disk
     can be a mix of thread A's and thread B's outputs (A wrote
     `.ttgir`, B overwrote `.cubin`, A overwrote `.ptx`, …). When
     `CompiledKernel.__init__` at line 426 then reads those files by
     path, it may assemble an asm dict whose stages come from two
     different compile runs.

     This is only a **semantic** hazard if Triton compilation is
     non-deterministic for fixed `(src, options, env, triton_version,
     backend_target)` — which is the cache key. Triton intends
     compilation to be deterministic for a fixed key, and the
     higher-level in-memory dedup in `JITFunction.device_caches` is the
     first line of defense against the same process reaching this race
     at all. Under the determinism assumption, "mixed" stage files are
     byte-identical to a single run and the `CompiledKernel` is
     correct.

     Non-determinism would be an architectural cache-correctness bug,
     not a Python data race on shared mutable state, and it cuts
     equally across processes — which this single-process audit cannot
     fully assess. CLAUDE.md explicitly excludes "filesystem atomic
     replace by itself" from this pass.
  6. **Reader protection during an in-progress racing overwrite.**
     A third thread C that calls `compile()` during A/B's race and
     sees a committed `__grp__` file from a *previous* successful
     build can enter the "cache hit" branch and construct a
     `CompiledKernel` whose file reads (via `Path(p).read_bytes()` at
     line 427) may be open-and-read against files that A or B is
     concurrently `os.replace`-ing. POSIX `os.replace` is an atomic
     rename that unlinks the old inode while preserving readability
     for any already-open file descriptor, and `read_bytes()` opens
     then immediately reads then closes — so C either sees the entire
     old content (file descriptor opened before the rename, rename
     unlinks the old inode, content still readable) or the entire new
     content (opened after the rename). No torn file read. Cross-file
     atomicity is again the determinism question from point 5.
  7. **knobs reads inside the cache path.** `knobs.compilation.override`
     (254), `dump_ir` (255), `store_binary_only` (256), `always_compile`
     (267), `use_ir_loc` (319) are each single attribute loads into a
     local. Concurrent mutation is a benign stale read and does not
     affect which on-disk files get written — the locals stay stable
     within the call. See also triage item #7.
- **Verdict: not a bug at the `compile()` level.** No issue file
  created. All shared-state coordination in this region is
  filesystem-level and relies on `os.replace` atomicity plus the
  group-file commit marker, both of which are correct. The only
  non-trivial residual hazard — mixed stage files under
  non-deterministic compile — is an architectural cache-correctness
  concern that belongs to the determinism audit, not the Python
  free-threading audit, and is upstream-protected by the in-memory
  `JITFunction.device_caches` dedup. `runtime/cache.py` itself still
  warrants the dedicated Pass 2 review (notably the dead `lock_path`,
  `RemoteCacheManager` / `Redis` backend races, and `triton_key`'s
  `@lru_cache`); that is out of scope here.

### 7. `compile()` backend selection and knobs reads

- **File/lines:** `compiler.py:234` (`make_backend`), `compiler.py:254-256`
  (`knobs.compilation.override`, `dump_ir`, `store_binary_only`),
  `compiler.py:267` (`knobs.compilation.always_compile`),
  `compiler.py:319` (`knobs.compilation.use_ir_loc`).
- **Deeper analysis:**
  1. **Single-load-into-local discipline.** Each of the five knobs
     reads is a single `knobs.compilation.X` attribute access stored
     straight into a function local (`enable_override`,
     `enable_ir_dump`, `store_only_binary`, `always_compile`,
     `use_ir_loc`). Every subsequent use inside `compile()` —
     `fn_override_manager = get_override_manager(...) if
     enable_override else None` at 257, `fn_dump_manager = ... if
     enable_ir_dump else None` at 258, `if not always_compile and
     metadata_path is not None:` at 268, `if (not store_only_binary)
     or (ext in ...)` at 338, `if ir_source and use_ir_loc:` at 320,
     `if use_ir_loc == ext:` at 346 — reads the local, never the
     descriptor again. So there is no TOCTOU on any of these within
     `compile()`: whichever value was observed at the top of the call
     is the value the rest of the call acts on. A concurrent
     mutation by another thread becomes visible on the *next* call,
     which is the intended behavior for these process-wide toggles.
  2. **Knob descriptor safety.** Each of these knobs is an
     `env_bool` / `env_opt_str` descriptor on `CompilationKnobs`
     (`knobs.py:357-363`). The descriptor `__get__` path is its own
     audit target (knobs.py pass), but even under the worst-case
     assumption that `__get__` races with a concurrent `__set__` and
     returns a stale or torn value, the consequence is contained:
     the local captures *some* valid value, and that value stays
     stable for the rest of the call. There is no compound
     check-then-act inside `compile()` that could be split across a
     descriptor race.
  3. **`make_backend(target)` global registry access.**
     `make_backend` (`compiler.py:366-371`) does
     `[x.compiler for x in backends.values() if
     x.compiler.supports_target(target)]` then calls `actives[0]
     (target)`. The `backends` dict is module-level state in
     `python/triton/backends/__init__.py:66`, populated exactly once
     by `_discover_backends()` at import time and **never mutated
     afterwards** — a grep across `python/triton` confirms the only
     writes are inside `_discover_backends` itself
     (`backends/__init__.py:53,61`). Module import is serialized by
     the CPython import lock, and the write-once-at-import pattern
     is explicitly excluded from reporting by CLAUDE.md. Concurrent
     `backends.values()` iteration is therefore iteration over a
     frozen dict and is safe.
  4. **Backend constructor side effects.** `actives[0](target)`
     constructs a fresh backend compiler instance per `compile()`
     call. The instance itself is call-local. Whether the
     constructor touches shared state (e.g. an `nvidia` backend
     reaching into `CudaUtils` or `driver.active`) is a
     backend-specific concern covered by the backend and
     `runtime/driver.py` passes, not by this file.
  5. **`knobs.compilation.listener` at line 227.** Read once into
     `compilation_listener` at the top of `compile()` and then reused
     at lines 271 and 324. Same single-load-into-local discipline as
     the other knobs. If a concurrent thread swaps the listener mid
     call, this call continues to use the listener that was
     installed when it entered — the intended semantics. **Note:**
     the listener *slot* on `CompilationKnobs` is the concern of a
     separate `knobs.py` writeup; `compile()`'s own usage is clean.
  6. **Double-read check.** The only knob in `compile()` that is
     read twice is `knobs.runtime.add_stages_inspection_hook` at
     lines 247-248, which is issue #2 (already written up). No
     other `knobs.*` attribute is loaded more than once per call.
- **Verdict: not a bug.** No issue file created. The knobs reads
  follow a consistent single-load-into-local pattern inside
  `compile()`, so concurrent mutation of a knob by another thread
  cannot split a decision within a single call. `make_backend` is
  safe because `backends` is a write-once import-time global and
  never mutated after `_discover_backends()` returns. The remaining
  concerns — descriptor-level races inside `knobs.py`, backend
  constructor shared-state touching, and the `add_stages_inspection_hook`
  double-read — are tracked in their own issue files or in the
  relevant upstream passes.

### 8. `IRSource.module` / `IRSource.__init__` context sharing

- **File/lines:** `compiler.py:87-119`, `compiler.py:235-240`,
  `compiler.py:299-307`. `IRSource.__init__` calls
  `ir.load_dialects(context)`, `backend.load_dialects(context)`, and
  `ir.parse_mlir_module(self.path, context)`, which stores a module
  that holds a reference to that context. `make_ir` later does
  `self.module.context = context` and returns `self.module`.
- **Deeper analysis:**
  1. **IRSource construction is gated to `compile()` only.** A
     codebase-wide grep for `IRSource(` finds three hits:
     `compiler.py:240` (inside `compile()`), and two unit tests
     (`test/unit/tools/test_irsource.py:44,89`). No production call
     site outside `compile()` constructs one. The class is also not
     re-exported at the package top level as a public API.
  2. **`compile()` blocks passing a pre-built `IRSource` by assert.**
     At `compiler.py:235-240`:
     ```python
     ir_source = not isinstance(src, ASTSource)
     if ir_source:
         assert isinstance(src, str), "source must be either AST or a filepath"
         context = ir.context()
         src = IRSource(src, context, backend)
     ```
     If a caller did pass a shared `IRSource` instance, `ir_source`
     would be True and the `assert isinstance(src, str)` would fail
     immediately. So even a user reaching for the "share an IRSource
     across threads" pattern is blocked at the API boundary — the
     only way into this code path is to hand `compile()` a string
     path, at which point the `IRSource` is freshly constructed on
     line 240 and never leaves the call.
  3. **Context lifetime is call-local on both branches.** For the
     `IRSource` branch, the fresh `context` from line 239 is the one
     stored inside `self.module` during `parse_mlir_module` at line
     107 and is the same object threaded back through to
     `src.make_ir(..., context)` at line 307 (line 299-302 skips
     re-creating `context` precisely *because* `IRSource.__init__`
     already used it). The `self.module.context = context`
     reassignment at line 118 therefore reassigns to the object
     the module was already parented to — it is a redundant
     self-assignment on the normal path, not a rebinding across
     contexts. For the `ASTSource` branch, `context` is created
     fresh at line 300 and used only for that call's `make_ir`.
     Either way, the context and the resulting `module` are locals
     of the `compile()` frame, reachable only through call-local
     references until `CompiledKernel.__init__` copies out the
     byte-level stage outputs.
  4. **No cross-call `IRSource` state.** `self.src`, `self.name`,
     `self.signature`, `self.ext`, `self.module`, `self.language`
     are all set once in `__init__` on a fresh per-call instance.
     None of them are mutated outside `__init__` other than
     `self.module.context =` in `make_ir`, and that method runs
     exactly once per `IRSource` in `compile()`.
  5. **`ir.context()` / `ir.load_dialects` / `parse_mlir_module`
     internals.** These calls cross into `python/src/ir.cc` and
     MLIR's C++ context machinery. MLIR `MLIRContext` is not
     thread-safe for concurrent mutation, and `load_dialects` /
     `parse_mlir_module` mutate the context. But because every
     `compile()` invocation owns its own fresh `ir.context()` and
     never publishes it to another thread, two concurrent
     `compile()` calls operate on two disjoint contexts. The
     context-internal thread-safety story is a Pass 3
     (`python/src/ir.cc`) topic; it is not raced by anything
     `compile()` does at the Python level.
  6. **Filesystem read during `__init__`.** `self.src =
     path.read_text()` at line 94 and `ir.parse_mlir_module(self.path,
     context)` at line 107 both open the on-disk IR file. If another
     thread or process is atomically replacing that file
     concurrently (see item #6 on `os.replace` semantics), the read
     is either the whole old file or the whole new file — no torn
     read. Not an IRSource-level concern.
- **Verdict: not a bug.** No issue file created. The hypothetical
  "shared `IRSource` across threads" scenario sketched in the
  original triage is blocked at the `compile()` API boundary by the
  `assert isinstance(src, str)` check, and every `IRSource` instance
  is therefore call-local with a call-local `ir.context()`. The
  `self.module.context = context` reassignment is a redundant
  self-assignment on the normal path, not a cross-context rebind.
  Context-internal thread safety and `python/src/ir.cc` Python C
  API concerns remain Pass 3 topics.

### 9. `code_generator.py` — module-level state and `gscope` iteration

Written up in
[code-generator-gscope-iteration-race.md](code-generator-gscope-iteration-race.md).

The initial triage focused on module-level caches and found none:
`_condition_types = {bool, int, type(None)}` at line 108 is written once
at import and never mutated; `CodeGenerator.builtin_namespace` at lines
342-350 is populated at class-definition time and never mutated; there
is no `@lru_cache`, no `@cached_property`, no `defaultdict` global;
`ast_to_ttir` constructs a fresh `CodeGenerator` per call; `module_map`
is read-only inside the file. Nothing to report on module-level state
per se.

Deeper analysis surfaced a distinct issue: `CodeGenerator.__init__`
seeds `self.gscope` by iterating `gscope.items()` at
`code_generator.py:309`, where `gscope` is returned by
`JITFunction.get_capture_scope()` at `runtime/jit.py:492-497`. In the
common no-closure case that function returns **`fn.__globals__` by
identity** — the live module dict — not a copy. Concurrent
module-level assignment in another thread while the compile thread is
iterating violates the "iteration under concurrent mutation" rule
(see `PYTHON_THREADSAFETY.md`). Undefined element sequence →
spurious `NameError` compile failures, or a constexpr global being
read *after* a rebind in thread B while `used_global_vals` / cache
key was computed against the old value, producing a stale-keyed
compiled kernel. The same iteration site runs again per callee at
`code_generator.py:1332`. Tier 1 Significant.

Fix is one line in `runtime/jit.py`: return `dict(self.__globals__)`
on the no-closure branch so `CodeGenerator.__init__` iterates a
call-local snapshot. The analogous iteration in
`JITFunction.cache_key` → `DependenciesFinder` at
`runtime/jit.py:509` should get the same treatment independently;
neither of the existing jit.py writeups covers it.

### 10. `ASTSource.hash()` — derived from `fn.cache_key`

- **File/lines:** `compiler.py:71-76`. Reads `self.fn.cache_key`.
- **Concern:** `cache_key` on `JITFunction` is protected by `_hash_lock`
  (covered in the jit.py audit, issue #11). `ASTSource` itself is
  constructed per-call by `JITFunction._do_compile`, so there is no
  `ASTSource`-level shared-state race. **Not a bug.**

### 11. `LazyDict` — per-call instance

- **File/lines:** `compiler.py:374-387`. Constructed inside
  `CompiledKernel.launch_metadata` at `compiler.py:486` and returned up
  the stack to the launch hooks via the launcher C call at
  `compiler.py:501-502`.
  ```python
  class LazyDict:
      def __init__(self, data):
          self.data = data
          self.extras = []

      def get(self):
          for func, args in self.extras:
              self.data = self.data | func(*args)
          self.extras.clear()
          return self.data

      def add(self, func, args):
          self.extras.append((func, args))
  ```
- **Deeper analysis:**
  1. **Construction site and lifetime.** `launch_metadata` is called
     from the `runner` closure returned by `CompiledKernel.__getitem__`
     (`compiler.py:493-504`). Each invocation of `runner(*args, ...)`
     calls `self.launch_metadata(grid, stream, *args)` (line 500),
     which constructs a fresh `LazyDict({"name": ..., "function": ...,
     "stream": ...})` on line 486. The LazyDict reference is held only
     in the `runner` frame's `launch_metadata` local until `self.run(
     ..., launch_metadata, launch_enter_hook, launch_exit_hook, ...)`
     returns at line 501-502. It never escapes that frame, and the
     frame belongs to the calling thread.
  2. **Tier 2 sharing check.** Two threads each calling
     `kernel[grid](...)` (or each calling a previously-captured
     `runner = kernel[grid]`) independently create their own
     `LazyDict` per call. Even if the same `CompiledKernel` instance
     is shared (which it is, via `JITFunction.device_caches`), the
     LazyDict itself is constructed inside the per-call
     `launch_metadata` method and is not stored on `self`. There is
     no shared `LazyDict` object across threads.
  3. **Hook chain reuse within one launch.** The same LazyDict
     instance is passed to every hook in `launch_enter_hook` and
     later to every hook in `launch_exit_hook` (both
     `HookChain[LaunchHook]`). Within that single launch frame,
     hooks execute sequentially on one thread (the C launcher does
     not spawn worker threads for hook dispatch). The first hook
     that calls `.get()` mutates `self.data = self.data | func(*args)`
     and `self.extras.clear()`; subsequent `.get()` calls from other
     hooks observe an empty `self.extras`, iterate zero times, and
     return the already-merged `self.data`. The mutation is
     idempotent for any number of `.get()` calls within the same
     launch, and the sequence is serialized by virtue of all hooks
     running on the calling thread.
  4. **Proton direct-`.data` reads are safe.** Proton's
     `LaunchHook.enter` reads `metadata.data.get("name")` directly
     at `proton/hooks/launch.py:94` before calling `metadata.get()`,
     and `InstrumentationHook.enter/exit` reads `metadata.data.get(
     "function")` and `metadata.data.get("stream")` at
     `proton/hooks/instrumentation.py:254-255,262-263`. These bypass
     `.get()` entirely and only touch the plain dict stored in
     `self.data`. Because the LazyDict is thread-local to the launch
     frame, there is no reader on another thread that can observe
     the `self.data = self.data | ...` rebind mid-flight. The reads
     are plain dict lookups on a dict that was either (a) the
     original `{"name": ..., "function": ..., "stream": ...}` dict
     constructed at line 486, or (b) the post-merge dict produced
     by an earlier hook's `.get()` call on the same thread. Both
     cases are deterministic on the calling thread.
  5. **`add()` contract.** `launch_metadata` calls `ret.add(
     self.src.fn.launch_metadata, (grid, self.metadata, arg_dict))`
     exactly once at line 490 before returning, and no other caller
     adds to this `LazyDict` after it leaves `launch_metadata`.
     The `.append` is on a call-local list reached from a call-local
     `LazyDict`; no concurrent add/iteration hazard.
  6. **`self.function` / `self.metadata` snapshot.** The LazyDict's
     initial `data` dict captures `self.function` at construction
     time (line 486). `_init_handles` is called at line 485 before
     the LazyDict is built, so `self.function` is expected to be
     populated. If `_init_handles` races per issue #1, a second
     thread could observe `self.function is None` at line 486 — but
     that is the #1 consequence, not a new LazyDict bug. The
     LazyDict itself stores whatever value was visible at line 486
     and propagates it to hooks unchanged.
  7. **Dict merge operator under free-threading.** `self.data |
     func(*args)` at line 382 invokes `dict.__or__`, which under
     free-threaded CPython acquires the critical section on each
     operand dict during the merge and returns a fresh dict. Since
     both operands are call-local here (the left is this LazyDict's
     own `self.data`, the right is whatever the user's
     `launch_metadata` function returned), there is no cross-thread
     dict involved, and even the container-level guarantee is not
     load-bearing.
- **Verdict: not a bug.** No issue file created. `LazyDict` is
  constructed per launch, mutated only on the calling thread, and
  never stored on a shared object. The `self.data` rebind in `.get()`
  is idempotent across multiple hook invocations within the same
  launch frame, and Proton's direct `metadata.data` reads are safe
  because no other thread can observe the mid-launch state of the
  same LazyDict. The downstream `HookChain` iteration-while-mutation
  hazard on the `launch_enter_hook` / `launch_exit_hook` slots is
  already covered by issue #3 — that is a hazard on the `HookChain`
  `.calls` list, not on the `LazyDict` argument threaded through it.

## Near-miss notes

- **`make_shared_mem` lru_cache thread-safety under free-threading.**
  We are treating this as a CPython-level guarantee (see item #4) but
  if the runtime/driver audit later turns up concerns about
  `driver.active.utils.get_device_properties` re-entrancy, revisit
  whether duplicate calls are actually benign.

- **`metadata_group` / `metadata` dict mutation inside `compile()`.**
  Both dicts are locals in `compile()` and do not escape until they
  are handed to `CompiledKernel.__init__`, which only reads them.
  No cross-thread sharing.

- **`filter_traceback` walking `e.__traceback__`.** Exception instances
  are not shared across threads under normal usage; no audit concern.
