# Upstream issue: segfault on Python 3.13t during Triton import

**Upstream link:** [triton-lang/triton#6721](https://github.com/triton-lang/triton/issues/6721)

- **Patch:** No standalone patch. The two in-scope contributing races are addressed by `driver-default-lazy-init-race.patch` (DriverConfig lazy-init lock) and `../nvidia-driver/cudautils-singleton-race.patch` (CudaUtils locked init-once). The single-thread native-extension hypotheses described below remain follow-ups outside `runtime/driver.py`.
- **Severity:** SEVERE
- **Tier:** 1

## Summary

User reports that importing Triton under **Python 3.13t** (the
free-threaded / no-GIL build) causes a hard segmentation fault rather
than either working or failing with a clear error. The crash happens
during `pytest` *collection* — before any test body runs — as a
side-effect of importing a test module that pulls in a Triton-decorated
kernel.

## Crash signature

- `Fatal Python error: Segmentation fault`
- Core dumped during module import.
- Stack trace originates inside
  `triton/backends/nvidia/driver.py` at line 71, in
  `compile_module_from_src`.
- The crash propagates up through an `@autotune` decorator applied at
  import time in the user's test module.

## Reproduction

1. Build or install a Python 3.13t (free-threaded) interpreter.
2. `pip install triton` (NVIDIA backend).
3. Run `pytest` against a test file that imports any module containing
   a `@triton.autotune`-decorated kernel.
4. Pytest crashes during the collection phase with the segfault above;
   no tests execute.

## User expectation

The user expects one of:

- Working free-threading support, or
- A clean `ImportError` / `RuntimeError` at import time stating that
  free-threading is unsupported.

A hard segfault with core dump is the unacceptable third outcome.

## Relevance to this audit

The crash site — `compile_module_from_src` inside the NVIDIA backend
driver — is the exact native helper invoked from
`CudaUtils.__init__` at
`third_party/nvidia/backend/driver.py:65-90`, which is itself reached
via the lazy-singleton path
`DriverConfig.default` → `_create_driver()` → `CudaDriver()` →
`CudaUtils()`.

That lazy-singleton path has two layered TOCTOU races under
free-threading:

1. **`DriverConfig.default` / `DriverConfig.active`** —
   `if self._default is None: self._default = _create_driver()` has
   no lock (`runtime/driver.py:32-34`). Two threads reaching
   `driver.active` concurrently can both enter `_create_driver()`.
   Written up in
   [driver-default-lazy-init-race.md](driver-default-lazy-init-race.md).
2. **`CudaUtils.__new__`** — `if not hasattr(cls, "instance"):
   cls.instance = super().__new__(cls)` is itself racy
   (`third_party/nvidia/backend/driver.py:60-63`), *and*
   `CudaUtils.__init__` calls `compile_module_from_src` and writes
   five `global` names (`PyCUtensorMap`, `PyKernelArg`,
   `ARG_CONSTEXPR`, `ARG_KERNEL`, `ARG_TUPLE`) with no
   synchronization. (Out of scope for `runtime/driver.py`; flagged
   here as the downstream target for a separate fix.)

If two Python threads enter the lazy bring-up simultaneously, two
`compile_module_from_src` calls run in parallel — each builds the
same `cuda_utils` extension, writes to the same on-disk build cache,
and `dlopen`s the same `.so`. That is a plausible proximate cause of
the reported segfault.

## Single-thread caveat

`pytest` collection is not obviously multi-threaded, so the reporter
may not actually have two Python threads racing in `driver.py`. Two
alternative hypotheses to keep in mind:

- The NVIDIA extension modules that `compile_module_from_src`
  produces may not declare free-threading compatibility
  (`Py_mod_gil = Py_MOD_GIL_NOT_USED` / `PyUnstable_Module_SetGIL`).
  CPython 3.13t then falls back to enabling the GIL for that module
  at import, which should be safe but has had known import-time
  bugs. A native segfault at import of a non-compatible extension is
  plausible even in single-threaded code.
- Something inside the NVIDIA backend's build or `dlopen` path may
  itself be unsafe under a free-threaded interpreter regardless of
  concurrency — e.g. a static initializer that calls Python C API
  without first attaching a Python thread state
  (`PyGILState_Ensure`), relying on GIL serialization that no
  longer exists under 3.13t.

Both hypotheses are outside `runtime/driver.py`. The audit here
addresses the *upstream* lazy-singleton races that would expose
these downstream problems to concurrent entry under normal
multi-threaded Triton use — i.e. even if #6721 turns out to be a
pure single-thread native bug, the races in this directory are
still real and would produce an equivalent crash under concurrent
first-use in production code.

## Status

Not fixed. Tracked in this audit directory as the motivating example
for the `runtime/driver.py` writeups. The minimum fix inside this
directory's scope is issues (1) and (2) in
[issues.md](issues.md); the downstream `CudaUtils` race and the
native-extension free-threading declarations are follow-ups outside
`runtime/driver.py`.
