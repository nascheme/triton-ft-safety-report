# `PyCUtensorMap` / `PyKernelArg` / `ARG_*` module globals lazy init race

- **Issue-Id:** FT037
- **Status:** Open
- **Rank:** Low
- **Component:** `third_party/nvidia/backend/driver.py`
- **Patch:** [`cudautils-singleton-race.patch`](cudautils-singleton-race.patch)

- **Shared state:** Five module-level globals — `PyCUtensorMap`,
  `PyKernelArg`, `ARG_CONSTEXPR`, `ARG_KERNEL`, `ARG_TUPLE` — initialized
  to `None` at import time, lazily populated from `CudaUtils.__init__`.
- **Writer(s):** `CudaUtils.__init__` — five `STORE_GLOBAL` statements,
  re-run on every `CudaUtils()` call because the broken singleton pattern
  does not suppress `__init__` (see
  [cudautils-singleton-race.md](cudautils-singleton-race.md)).
- **Reader(s):** `annotate_arguments()`, called from
  `CudaLauncher.__init__`.
- **Race scenario:** The only path to `CudaLauncher` goes through
  `CudaDriver.__init__`, which calls `CudaUtils()` synchronously. That
  call blocks in `compile_module_from_src` until the module is built,
  then assigns all five globals before returning. Therefore any thread
  that has a `CudaDriver` reference has already completed a full
  `CudaUtils.__init__` pass and the globals are set.

  The residual race is: Thread A is re-running `CudaUtils.__init__` (due
  to the broken singleton) and is mid-way through overwriting the five
  globals.  Thread B reads a mix of old and new values.  However, every
  `compile_module_from_src` call for the same inputs produces functionally
  equivalent results, so old and new values are interchangeable.  The
  consequence is benign.
- **Suggested fix:** Fix the singleton pattern
  ([cudautils-singleton-race.md](cudautils-singleton-race.md)) so that
  `__init__` runs only once. The globals then become write-once and the
  race disappears. Ideally, move them onto the singleton instance so they
  are not module-level mutable state at all.
