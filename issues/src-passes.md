# python/src/passes.cc Free-Threading Issues

Scope: pybind11 bindings for MLIR pass-pipeline builder functions and two
analysis wrappers.

No individual issue files are warranted. Most risk routes through caller-owned
`MLIRContext` / `PassManager` objects already summarized in [`ir.md`](ir.md).

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| 1 | Low | `ModuleAllocation` / `ModuleMembarAnalysis` bindings | Latent only: would mutate caller context, but shipping paths use per-compile contexts and no Python callers |
| 2 | Low | `init_plugin_passes` -> `loadPlugins()` | Plain-bool TOCTOU shape, but module-init callers pre-warm under the import lock |
| 3 | Low | Pass-builder lambdas | Sharing one `PassManager` across threads is a caller-contract race |
| 4 | Low | Pass-builder factory functions | Spot-check only if factory-local static state appears |
| 5 | Rejected | `m.def(...)` / `py::class_` registrations | Import-lock write-once module registration |

## Notes

- The file owns no important file-scope mutable state.
- The analysis wrappers have no Python callers in the shipping tree today.
- Revisit #3 if Triton starts exposing reusable `PassManager` objects as a
  supported Python API.
