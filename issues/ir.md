# python/src/ir.cc Free-Threading Issues

Scope: the pybind11 surface for `mlir::MLIRContext`, operation builders,
environment helpers, and `PassManager::run`.

No individual issue files are warranted at this time. The initially scary
context-sharing candidates were downgraded after verifying that shipping
compile paths construct fresh `MLIRContext` / `PassManager` objects per compile.

## Issues

| # | Severity | Tier | Component | Issue |
|---|----------|------|-----------|-------|
| 1 | LOW | 2 | `PassManager::run` debug flags | `::llvm::DebugFlag` / `setCurrentDebugTypes` mutation with GIL released; diagnostic-only |
| 2 | LOW | 2 | `PassManager::run` context mutation | `context->disableMultithreading()` is latent only because contexts are per compile |
| 3 | LOW | 2 | `load_dialects` | Latent only: unsafe on a shared `MLIRContext`, not shared in shipping paths |
| 4 | LOW | 2 | Operation builders | Latent only: op/type/attr construction would race on a shared context |
| 5 | LOW | 1 | `plugin::loadPlugins()` | Plain-bool TOCTOU shape, but reached at module init under the import lock today |
| 6 | LOW | 2 | `PassManager::run` options | Per-call mutations are on per-compile objects; reproducer file write is debug-only |
| 7 | MED | 3 | `setupTritonDiagnosticHandler` | Diagnostic output goes to process-wide `llvm::errs()` |
| 8 | LOW | 3 | Env helpers | POSIX `getenv` is unsafe against concurrent `setenv` |

## Notes

- The durable invariant is "contexts are per compile." Re-check #2-#4 if Triton
  ever pools or caches `MLIRContext` objects.
- #1 and #7 affect diagnostics, not generated code.
- #5 is the same lazy-native-init pattern as `specialize.cc`, but current import
  ordering prevents user-thread reachability.
