# third_party/nvidia/backend/driver.py Free-Threading Issues

Scope: NVIDIA backend driver bring-up via `CudaDriver()` and `CudaUtils()`.
This is downstream of [`runtime-driver.md`](runtime-driver.md)'s default-driver
lazy initialization race.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT037 | Low | module globals | `CudaUtils()` rewrites `PyCUtensorMap`, `PyKernelArg`, and `ARG_*` module globals on each construction. Normal launch paths only use them after construction completes, so the concern is low-priority repeated initialization. |
| FT036 | Low | `CudaUtils` singleton | The `CudaUtils` singleton check is not synchronized, so concurrent driver bring-up can duplicate `compile_module_from_src` / `dlopen` work. This is usually triggered through FT040 and is not the root driver race. |
| 3 | Low | `CudaDriver.__init__` | Every `CudaDriver()` calls `CudaUtils()`; covered by FT036/FT037 |

## Notes

- The scary "globals still `None`" path was downgraded: normal code reaches
  `CudaLauncher` only after `CudaUtils()` completes.
- Remaining risk is duplicate native-module build/load during concurrent driver
  construction, usually triggered by FT040 in `runtime-driver`.
- Relationship to [`runtime-driver/seg-fault-gh-6721.md`](runtime-driver/seg-fault-gh-6721.md)
  is plausible but not proven.
