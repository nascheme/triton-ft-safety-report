# third_party/nvidia/backend/driver.py Free-Threading Issues

Scope: NVIDIA backend driver bring-up via `CudaDriver()` and `CudaUtils()`.
This is downstream of [`runtime-driver.md`](runtime-driver.md)'s default-driver
lazy initialization race.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| FT037 | LOW | module globals | 1 | [`PyCUtensorMap` / `PyKernelArg` / `ARG_*` module globals rewritten on every `CudaUtils()` call](nvidia-driver/module-globals-lazy-init-race.md) |
| FT036 | LOW | `CudaUtils` singleton | 1 | [Broken singleton pattern causes duplicate `compile_module_from_src` / `dlopen` under concurrent driver bring-up](nvidia-driver/cudautils-singleton-race.md) |
| 3 | LOW | `CudaDriver.__init__` | 1 | Every `CudaDriver()` calls `CudaUtils()`; covered by FT036/FT037 |

## Notes

- The scary "globals still `None`" path was downgraded: normal code reaches
  `CudaLauncher` only after `CudaUtils()` completes.
- Remaining risk is duplicate native-module build/load during concurrent driver
  construction, usually triggered by FT040 in `runtime-driver`.
- Relationship to [`runtime-driver/seg-fault-gh-6721.md`](runtime-driver/seg-fault-gh-6721.md)
  is plausible but not proven.
