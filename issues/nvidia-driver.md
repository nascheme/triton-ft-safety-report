# third_party/nvidia/backend/driver.py Free-Threading Issues

Scope: NVIDIA backend driver bring-up via `CudaDriver()` and `CudaUtils()`.
This is downstream of [`runtime-driver.md`](runtime-driver.md)'s default-driver
lazy initialization race.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT037 | Low | module globals | [`PyCUtensorMap` / `PyKernelArg` / `ARG_*` module globals rewritten on every `CudaUtils()` call](nvidia-driver/module-globals-lazy-init-race.md) |
| FT036 | Low | `CudaUtils` singleton | [Broken singleton pattern causes duplicate `compile_module_from_src` / `dlopen` under concurrent driver bring-up](nvidia-driver/cudautils-singleton-race.md) |
| 3 | Low | `CudaDriver.__init__` | Every `CudaDriver()` calls `CudaUtils()`; covered by FT036/FT037 |

## Notes

- The scary "globals still `None`" path was downgraded: normal code reaches
  `CudaLauncher` only after `CudaUtils()` completes.
- Remaining risk is duplicate native-module build/load during concurrent driver
  construction, usually triggered by FT040 in `runtime-driver`.
- Relationship to [`runtime-driver/seg-fault-gh-6721.md`](runtime-driver/seg-fault-gh-6721.md)
  is plausible but not proven.
