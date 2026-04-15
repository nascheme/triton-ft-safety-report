# runtime/driver.py Free-Threading Issues

Issues in `python/triton/runtime/driver.py` affecting free-threaded Python 3.14t.

File in scope:

- `runtime/driver.py` — `DriverConfig` lazy-singleton holding the active /
  default `DriverBase` instance. Module-level object `driver = DriverConfig()`
  is read as `driver.active` from every hot path in the project (compiler,
  JIT dispatch, autotuner, testing, tools).

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | Significant | DriverConfig.default | 1 | [`DriverConfig.default` lazy init race creates duplicate driver instances](driver-default-lazy-init-race.md) |
| 2 | Significant | DriverConfig.active  | 3 | [`DriverConfig.active` lazy init / `set_active` ordering race](driver-active-set-active-race.md) |
| 3 | Significant | hot-path callers     | 3 | [Hot-path callers re-read `driver.active` across multiple statements](driver-active-caller-reread.md) |

See also: [seg-fault-gh-6721.md](seg-fault-gh-6721.md) — upstream segfault
report under Python 3.13t that is a plausible downstream consequence of
issue #1.

## Triage notes

`runtime/driver.py` is tiny (50 lines) but sits directly in CLAUDE.md's
"global registries and lazy singletons on the hot path" bucket
(`runtime.driver.driver`). Everything of interest is on the `DriverConfig`
object that is instantiated once at import time and mutated lazily
afterwards.

### 1. `DriverConfig.default` — lazy init TOCTOU

- **Shared state:** `self._default` on the module-level `driver`
  (`driver.py:27, 32-34`), starts `None` and is populated on first access
  to the `default` property.
- **Writer:** `default` getter at `driver.py:32-34`:
  ```python
  if self._default is None:
      self._default = _create_driver()
  return self._default
  ```
- **Reader:** `default` getter itself, plus `active` getter
  (`driver.py:38-39`) which calls `self.default` when `self._active is
  None`.
- **Race scenario:** Thread A calls `driver.default` (or transitively via
  `driver.active`). It reads `self._default is None`, enters
  `_create_driver()` which in turn instantiates the concrete backend
  driver class. For the NVIDIA backend that constructor compiles and
  `dlopen`s a native utils module via `compile_module_from_src` and
  initializes CUDA runtime state. Thread B arrives before A has written
  back `self._default`, also sees `None`, and also calls `_create_driver()`
  concurrently. Two independent driver instances are constructed in
  parallel; one gets overwritten.
- **Concrete consequences:**
  1. Duplicate native initialization. `_create_driver()` → concrete
     driver `__init__` → `compile_module_from_src` duplicates the
     filesystem build / `dlopen` sequence in two threads simultaneously,
     hitting any races inside that path (cache dir creation, build
     directory staging, `dlopen` of the same `.so`, lazy CUDA context
     init). This is a plausible root cause of user-visible segfaults
     reported during Triton import on free-threaded Python such as
     [triton-lang/triton#6721](https://github.com/triton-lang/triton/issues/6721)
     (see [seg-fault-gh-6721.md](seg-fault-gh-6721.md)).
  2. Two readers of `driver.default` / `driver.active` in the racing
     window can see *different* `DriverBase` instances. Downstream
     code (`compiler.py:455`, `compiler.py:468`, `jit.py:713-714`)
     caches handles and launcher objects keyed off a specific driver
     instance, so state populated under one instance is observed via
     the other. One instance is effectively leaked, with whatever
     native resources it holds.
  3. The `default` getter is not idempotent under a race: the final
     `self._default` observed by later callers is whichever thread wrote
     last. If anything in `_create_driver()` has side effects that are
     not safe to run twice, those fire twice.
  4. Partially-initialized `CudaUtils` instance. If both threads reach
     `CudaUtils.__new__` and one sees the in-flight `cls.instance`
     written by the other, both run `__init__` on the same object
     concurrently, writing ~eight attributes in an interleaved order.
     Readers that retrieve `driver.active.utils` mid-init can observe an
     object with some attributes set and others missing, giving
     `AttributeError` on the launch path.
  5. `CudaUtils.__init__` writes five `global` names
     (`PyCUtensorMap`, `PyKernelArg`, `ARG_CONSTEXPR`, `ARG_KERNEL`,
     `ARG_TUPLE`) in five separate statements. A concurrent reader can
     see some names updated and others still `None`.
- **Why the GIL previously hid this:** Under the GIL, the two-step
  `if self._default is None: self._default = _create_driver()` is still
  not atomic — a thread switch can happen during `_create_driver()`.
  In practice `_create_driver()` spends most of its time in native code
  where the GIL is released for I/O, but no other Python thread
  typically tries to enter `driver.default` at the same moment.
  Free-threading removes that accidental serialization.
- **Tier:** 1 (driver bring-up happens during normal import-time
  decoration and early launch; two unrelated threads reaching Triton at
  the same time is sufficient, they do not even have to share a kernel).
- **Suggested fix:**
  ```python
  import threading

  class DriverConfig:
      def __init__(self) -> None:
          self._default: DriverBase | None = None
          self._active: DriverBase | None = None
          self._lock = threading.Lock()

      @property
      def default(self) -> DriverBase:
          if self._default is None:
              with self._lock:
                  if self._default is None:
                      self._default = _create_driver()
          return self._default
  ```
  Fixing `DriverConfig` alone does not eliminate the downstream
  `CudaUtils.__new__` TOCTOU — that backend-internal race would need its
  own fix — but it prevents concurrent entry into `_create_driver()`.

### 2. `DriverConfig.active` — lazy init + `set_active` ordering

- **Shared state:** `self._active` (`driver.py:28, 37-46`).
- **Writers:**
  - `active` getter lazy init: `self._active = self.default`
    (`driver.py:38-39`).
  - `set_active(driver)` (`driver.py:42-43`).
  - `reset_active()` (`driver.py:45-46`), which reads `self.default`
    then writes `self._active`.
- **Readers:** Every call site that reads `driver.active` — `jit.py:676`,
  `jit.py:713-714`, `jit.py:809`, `compiler/compiler.py:135,232,453,
  455,468,470,498,499`, `autotuner.py:125,185`, plus tools/testing. This
  is the hottest shared object in the project.
- **Race scenarios:**
  1. *Lazy init vs concurrent reader.* Thread A in `active` reads
     `self._active is None` and begins evaluating `self.default`, which
     (per issue #1) may itself race. Thread B reading `driver.active`
     at the same moment can either observe the still-`None` slot (and
     kick off its own `self.default` evaluation) or observe a
     freshly-installed instance.
  2. *Concurrent `set_active` vs lazy init.* Thread A is about to
     execute `self._active = self.default`. Thread B calls
     `driver.set_active(custom_driver)`. Thread A then clobbers
     `self._active` back to `self.default`, silently losing the user's
     `set_active`. Wrong-driver selection on a hot path.
  3. *`reset_active` re-enters lazy `default` init.* `reset_active`
     does `self._active = self.default`. `self.default` is a property
     that may trigger `_create_driver()` (issue #1). Concurrent
     callers of `reset_active` can re-enter `_create_driver()` in
     parallel.
- **Why the GIL previously hid this:** Under the GIL, scenario A
  requires a Python-level thread switch between two adjacent statements,
  which is rare. Scenarios B and C rely on the `default` race. In
  practice production workloads bring up the driver once from a single
  thread. Under free-threading the window opens unconditionally.
- **Tier:** 3 (explicit config mutation via `set_active` /
  `reset_active`) plus Tier 1 (lazy init of the active slot on first
  use).
- **Suggested fix:**
  ```python
  @property
  def active(self) -> DriverBase:
      active = self._active
      if active is not None:
          return active
      with self._lock:
          if self._active is None:
              self._active = self.default   # self.default already serialized
          return self._active

  def set_active(self, driver: DriverBase) -> None:
      with self._lock:
          self._active = driver

  def reset_active(self) -> None:
      with self._lock:
          self._active = self.default
  ```
  `set_active` and `reset_active` are public API not called from within
  the Triton tree, so the expected users are out-of-tree backends,
  plugins, and test harnesses.

### 3. Hot-path callers re-read `driver.active`

- **Shared state:** `DriverConfig._active`. Locking `DriverConfig`
  internally (issue #2) makes individual `driver.active` reads and
  `set_active` writes consistent, but does not protect a caller that
  reads `driver.active` twice: each read is individually consistent,
  but two reads can see different drivers.
- **Key call sites:**
  - `CompiledKernel._init_handles` (`compiler.py:453-470`) reads
    `driver.active` four times in close succession (device,
    launcher_cls, load_binary, warp_size).
  - `runtime/jit.py:713-714` — `get_current_device()` then
    `get_current_stream(device)`.
  - `runtime/jit.py:809` — a second `get_current_device()` later in
    the same flow, expected to agree with line 713.
  - `runtime/autotuner.py:125,185` — `get_benchmarker()` and
    `get_current_target()` on separate statements.
- **Race scenario:** Thread A enters `_init_handles`, reads
  `driver.active` → `D1` for device and launcher_cls. Thread B calls
  `driver.set_active(D2)`. Thread A's next read at line 468 returns
  `D2`. It calls `D2.utils.load_binary(...)` and stores the result on a
  `CompiledKernel` whose `self._run` was bound to `D1.launcher_cls`.
  `CompiledKernel.run` then hands a `D2`-loaded handle to a `D1`-bound
  launcher. For unrelated backends, the native launch call receives a
  handle from the wrong address space.
- **Relationship to other issues:** Independent of issues #1 and #2 —
  even after the lazy-init races are closed, `set_active` is still a
  legitimate API and can fire between any two `driver.active` reads.
  Compounds
  [`compiler/compiled-kernel-init-handles-race.md`](../compiler/compiled-kernel-init-handles-race.md)
  — a separate lazy-init race on `CompiledKernel._run` /
  `CompiledKernel.function` — via an additional mechanism by which the
  two halves of `_init_handles` disagree on the target driver.
- **Why the GIL previously hid this:** Under the GIL, a thread switch
  inside a single Python function between two attribute loads is
  technically possible but rare (no I/O and no C-level GIL release
  points between reads). Under free-threading the two reads can
  interleave with a `set_active` unconditionally.
- **Suggested fix:** Cache `driver.active` into a local at the top of
  each affected function:
  ```python
  # compiler.py:453-470 after fix
  drv = driver.active
  device = drv.get_current_device()
  self._run = drv.launcher_cls(self.src, self.metadata)
  ...
  self.module, self.function, ... = drv.utils.load_binary(
      self.name, self.kernel, self.shared, device
  )
  warp_size = drv.get_current_target().warp_size
  ```
  The full set of sites can be found with
  `rg 'driver\.active' python/triton/{compiler,runtime,language}`.
  Functions that read `driver.active` only once do not need changes.

## Not worth reporting

- **Module-level `driver = DriverConfig()`** (`driver.py:49`). Written
  once at import time under the import lock, never replaced. CLAUDE.md's
  carve-out for write-once globals applies.

- **`backends` dict from `triton.backends`** (imported at
  `driver.py:5`). Populated once at import time by
  `_discover_backends()` and never mutated after. Read-only shared
  state; not a bug on its own. If any individual backend's driver
  `__init__` mutates process-global C++ state that is the backend's
  problem, not `driver.py`'s.

- **`_create_driver()` reading `os.environ`** (`driver.py:9`).
  `os.environ.get` is a single dict lookup; concurrent readers are
  safe. Mutation of `TRITON_DEFAULT_BACKEND` at runtime is not a
  supported workflow.

- **`DriverConfig.default` property returning a stale-but-consistent
  instance after the race window closes.** Once `self._default` has
  been written, subsequent readers all observe the same instance.
  Only the initial race window (issue #1) is a real hazard.
