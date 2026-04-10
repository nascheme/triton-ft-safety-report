# third_party/nvidia/backend/driver.py Free-Threading Issues

Issues in `third_party/nvidia/backend/driver.py`, the
NVIDIA backend entry point reached via
`runtime.driver.driver.active` → `CudaDriver()` → `CudaUtils()`. This is
the downstream half of the lazy-singleton story started in
[`runtime-driver/`](../runtime-driver/README.md) and the proximate crash
site for [triton-lang/triton#6721](../runtime-driver/seg-fault-gh-6721.md).

File in scope:

- `triton/third_party/nvidia/backend/driver.py`

Related out-of-scope file (referenced but not audited here):

- `triton/python/triton/runtime/build.py` — host of `compile_module_from_src`,
  the native-extension build/dlopen helper invoked from `CudaUtils.__init__`.

## Issues

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| 1 | Minor       | module globals         | [`PyCUtensorMap` / `PyKernelArg` / `ARG_*` module globals rewritten on every `CudaUtils()` call, but values are equivalent — benign staleness only](module-globals-lazy-init-race.md) |
| 2 | Minor       | `CudaUtils` singleton  | [Broken singleton pattern: `__init__` re-runs on every call, causing duplicate `compile_module_from_src` and `dlopen` under concurrent driver bring-up](cudautils-singleton-race.md) |
| 3 | Minor       | `CudaDriver.__init__`  | Every `CudaDriver()` instantiation triggers the full `CudaUtils()` bring-up via `self.utils = CudaUtils()` (line 336). Under the `DriverConfig.default` race in `runtime-driver/` issue #1, two concurrent default-driver constructions therefore double up the entire native init. No separate writeup; this is covered by issues #1 and #2 above. |

## Triage notes

`third_party/nvidia/backend/driver.py` is the NVIDIA backend driver wrapper
selected by `runtime.driver.DriverConfig._create_driver()`. It defines three
objects of interest:

- `CudaUtils` — a "singleton" wrapping the compiled `cuda_utils` native
  extension produced by `compile_module_from_src`. Its `__init__` both
  populates instance attributes (`self.load_binary`, `self.launch`, …) and
  writes five module-level globals (`PyCUtensorMap`, `PyKernelArg`,
  `ARG_CONSTEXPR`, `ARG_KERNEL`, `ARG_TUPLE`).
- `CudaLauncher` — per-`CompiledKernel` object. Reads
  `triton.runtime.driver.active.utils.launch` / `.build_signature_metadata` /
  `.fill_tma_descriptor_tiled` in its constructor and in the launch path.
- `CudaDriver` — the backend's `DriverBase` subclass. Its `__init__` calls
  `CudaUtils()` unconditionally, so *every* `CudaDriver` instance
  reconstructs the native module.

### 1. Module-level globals populated from `CudaUtils.__init__`

- **Shared state:** `PyCUtensorMap`, `PyKernelArg`, `ARG_CONSTEXPR`,
  `ARG_KERNEL`, `ARG_TUPLE` (lines 17–21). Initialized to `None`.
- **Writers:** `CudaUtils.__init__` (lines 73–82). Rewritten on *every*
  `CudaUtils()` call (see issue #2).
- **Readers:** `annotate_arguments()` (lines 194–198) reads `PyKernelArg`,
  `ARG_TUPLE`, `ARG_KERNEL`, `ARG_CONSTEXPR` unconditionally. Called from
  `CudaLauncher.__init__` at line 302, which runs when *any*
  `CompiledKernel` is materialized — i.e. on every compile-time or
  load-from-cache path, on every thread.
- **Why the original "globals still None" scenario is not reachable:**
  The only code path to `CudaLauncher` goes through `CudaDriver.__init__`,
  which calls `CudaUtils()` synchronously (line 336). That call blocks in
  `compile_module_from_src` until the module is built, then assigns all
  five globals before `CudaDriver.__init__` returns. A thread cannot have
  a `CudaDriver` reference (and therefore cannot reach `CudaLauncher`)
  without having already completed a full `CudaUtils.__init__` pass.
  The `CudaUtils.__new__` publishes `cls.instance` before `__init__` runs,
  but `CudaDriver.__init__` blocks on the full `CudaUtils()` call
  (including `__init__`), so this does not create a window where the
  globals are `None`.
- **Residual race:** Because the singleton pattern is broken (issue #2),
  `__init__` re-runs on every `CudaUtils()` call. Thread A could be
  mid-way through overwriting the five globals while Thread B reads them.
  However, every `compile_module_from_src` call for the same deterministic
  inputs produces functionally equivalent results, so old and new values
  are interchangeable. The consequence is benign.
- **Tier:** 2 (concurrent `CudaDriver` construction).
- **Triage:** Downgraded from SEVERE to Minor. The original scenario
  assumed a thread could reach `annotate_arguments` with `None` globals,
  but this is not reachable through normal code paths. The residual race
  involves equivalent values only. Written up in
  [module-globals-lazy-init-race.md](module-globals-lazy-init-race.md).

### 2. `CudaUtils` broken singleton pattern

- **Shared state:** `CudaUtils.instance` class attribute (line 62); plus all
  the instance attributes assigned in `__init__`
  (`self.load_binary`, `self.launch`, …, lines 83–90); plus the five module
  globals from issue #1.
- **The `__new__` TOCTOU:** Two threads arriving simultaneously can both
  observe `not hasattr(cls, "instance")` and both allocate; one
  `cls.instance` assignment wins. However, `cls.instance` is set to a bare
  (uninitialized) object — `__init__` has not yet run — so the losing
  allocation is simply discarded. This is wasteful but not harmful.
- **The real problem — `__init__` re-runs every call:** `type.__call__`
  always invokes `__init__` on the object returned by `__new__`, regardless
  of whether it was freshly allocated or cached. So *every* `CudaUtils()`
  call re-runs `__init__`, which re-runs `compile_module_from_src`,
  re-imports the compiled extension, and rewrites the module globals and
  instance attributes. The `# TODO: make static` comment on
  `self.utils = CudaUtils()` at line 336 shows the author was aware of
  this smell.
- **Race scenario:** Any concurrent `CudaDriver()` construction lands two
  `CudaUtils()` calls in parallel. Both run `compile_module_from_src` to
  completion, each `dlopen`ing the resulting `.so`.
- **Concrete consequence:**
  1. **Duplicate compile + dlopen of the same native extension.** Wasted
     work; also exposes any non-thread-safe state inside
     `compile_module_from_src` or the extension's static initializers to
     concurrent entry.
  2. **Mixed instance attributes are benign.** Although two threads
     overwrite `self.launch` etc. with last-writer-wins semantics, both
     threads produce functionally equivalent values from the same
     deterministic inputs. A reader observing `launch` from compile A and
     `build_signature_metadata` from compile B gets correct behavior from
     both.
  3. **Duplicate cache writes.** `compile_module_from_src` calls
     `cache.put(...)` concurrently on the same key; this is the cache
     layer's problem, but the broken singleton is the trigger.
- **Tier:** 2 (concurrent `CudaDriver` construction).
- **Triage:** Minor — the duplicate work produces equivalent results from
  deterministic inputs, so the consequence is wasted computation, not
  corruption. Still worth fixing as it's the root cause of issue #1 and
  eliminates an entire class of potential problems. Written up in
  [cudautils-singleton-race.md](cudautils-singleton-race.md).

### 3. `CudaDriver.__init__` does not guard `self.utils = CudaUtils()`

- **Shared state:** the singleton state inside `CudaUtils` (see #2).
- **Writer/reader:** `CudaDriver.__init__` (line 336). Every driver
  instance unconditionally calls `CudaUtils()`, even if one has already
  been built. The `# TODO: make static` comment flags this.
- **Race scenario:** Covered entirely by issues #1 and #2. Not a separate
  bug; rather, this line is the *trigger* that turns the issue #1 and #2
  hazards into a real race whenever two threads build a `CudaDriver`
  concurrently, which happens under `runtime-driver/` issue #1.
- **Triage:** Minor. No separate writeup. Fix is to make `utils` a class
  attribute or lazy-cached singleton populated under a lock.

## Relationship to the gh-6721 segfault

The story threads together as:

1. Two Python threads reach `triton.runtime.driver.active` for the first
   time with no GIL to serialize them. (`runtime-driver/` issue #1 / #2.)
2. Both enter `_create_driver()` → `CudaDriver()` → `CudaUtils()`.
3. Because `CudaUtils` uses a broken singleton pattern (issue #2), both
   threads run `compile_module_from_src` to completion and both `dlopen`
   the produced extension. Any non-thread-safe state inside the cache
   layer or the extension's static initializers is exposed to concurrent
   entry.

Whether this chain is actually what crashes gh-6721 is unverified.
The issue reporter's pytest collection path is not obviously
multi-threaded, so the true proximate cause may be a single-threaded
problem inside `compile_module_from_src`, inside the `cuda_utils`
extension's own initialization, or inside CPython's free-threaded
import machinery when it handles a non-declared extension — none of
which are in the scope of this file.

The minimum fix inside *this* file is to replace the broken singleton
with a proper locked-init-once pattern (see issue #2 suggested fix).
This eliminates the duplicate `compile_module_from_src` calls, the
duplicate `dlopen`, and the globals rewrite race in one change.

The `runtime-driver/` fixes are necessary separately; they are not in
scope for this file. Declaring `Py_mod_gil = Py_MOD_GIL_NOT_USED` on
the generated `cuda_utils` extension would silence the free-threaded
import warning and avoid the process-wide GIL re-enable, but that is
orthogonal to the races documented here and its effect on gh-6721 is
unverified.

## Not worth reporting

- **`dirname`, `include_dirs`, `libdevice_dir`, `libraries`** (lines 13–16).
  Set once at import time, never mutated. Import-lock protected.
- **`@functools.lru_cache` on `libcuda_dirs` / `library_dirs`** (lines 24,
  48). Free-threaded CPython's `functools.lru_cache` is internally
  thread-safe; worst case is that the `ldconfig` subprocess runs twice on
  first use. Benign duplicate work.
- **`ty_to_cpp`'s inline dict literal** (lines 103–120). Rebuilt on every
  call; not shared state.
- **`TMA_DTYPE_DEVICE_TO_HOST` / `TMA_TF32`** (lines 203–207). Written
  once at import time, read-only afterward.
- **`expand_signature`, `make_kernel_signature`, `make_tensordesc_arg`,
  `wrap_handle_tensordesc`**. Pure functions over their arguments; the
  only shared state they touch is `triton.runtime.driver.active.utils.*`,
  which is covered by the `runtime-driver/` writeups and by issues #1/#2.
- **`CudaLauncher` instance attributes.** Written in `__init__` on a
  per-kernel object; not shared between kernels.
- **`CudaDriver.get_current_target` / `get_active_torch_device` / etc.**
  Read-only wrappers over `torch.cuda` calls; any thread-safety concerns
  are torch's problem, not Triton's.
- **Module-level `import torch` inside `GPUDriver.__init__`** (base class,
  `backends/driver.py:54`). Python's import lock serializes first import;
  subsequent constructions re-bind local names only.
