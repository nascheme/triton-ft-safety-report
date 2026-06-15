# python/src/llvm.cc Free-Threading Issues

Scope: pybind11 bindings for LLVM/MLIR translation, module optimization,
object/assembly emission, diagnostics, and LLVM process-global options.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT028 | Low | `dumpSchedulingDAG` stderr redirection | `dumpSchedulingDAG` temporarily redirects process-wide `stderr`, so concurrent diagnostics can interleave or land in the wrong file. This is debug-output corruption, not ordinary codegen corruption. |
| FT030 | Low | `llvm::TimePassesIsEnabled` / `llvm::TimePassesPerRun` | Timing-enabled LLVM translation mutates and drains process-global timing state from GIL-released regions. The impact is scrambled timing reports, not compiled-code correctness. |
| 4 | Low | `init_targets` | Unconditional write to `llvm::parallel::strategy`; concurrent writers set the same value |
| 5 | Low | `to_module` | Unsafe on a shared `mlir::ModuleOp` / `LLVMContext`, but shipping paths use per-compile objects |
| 6 | Low | `optimize_module` | Unsafe on a shared `llvm::Module`, but shipping paths use per-compile modules |
| 7 | Low | LLVM module mutation bindings | Unsafe on a shared module/function object; caller contract today |
| 8 | Low | `init_triton_stacktrace_hook` | Process-global signal handler registration |

## Notes

- FT028 and FT030 are debug/diagnostic features. They can corrupt output or
  file descriptors, but they do not change ordinary codegen.
- Items #5-#7 matter only if Python callers share LLVM/MLIR objects across
  threads. Current compile paths do not.
