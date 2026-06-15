# python/src/llvm.cc Free-Threading Issues

Scope: pybind11 bindings for LLVM/MLIR translation, module optimization,
object/assembly emission, diagnostics, and LLVM process-global options.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT028 | Low | `dumpSchedulingDAG` stderr redirection | [Process-wide `dup` / `freopen(stderr)` / `dup2` sequence races with other threads and Python I/O](llvm/dump-sched-dag-stderr-redirection.md) |
| FT030 | Low | `llvm::TimePassesIsEnabled` / `llvm::TimePassesPerRun` | [Process-global timing flags used from GIL-released `translateLLVMIRToASM`](llvm/time-passes-global-flags.md) |
| 4 | Low | `init_targets` | Unconditional write to `llvm::parallel::strategy`; concurrent writers set the same value |
| 5 | Low | `to_module` | Unsafe on a shared `mlir::ModuleOp` / `LLVMContext`, but shipping paths use per-compile objects |
| 6 | Low | `optimize_module` | Unsafe on a shared `llvm::Module`, but shipping paths use per-compile modules |
| 7 | Low | LLVM module mutation bindings | Unsafe on a shared module/function object; caller contract today |
| 8 | Low | `init_triton_stacktrace_hook` | Process-global signal handler registration |

## Notes

- Runtime mutation of LLVM `cl::opt` is a pre-existing maintainer-rule
  violation, not a current-goal ask; it is summarized in
  [Dropped And Out-Of-Scope Notes](README.md#dropped-and-out-of-scope-notes).
- FT028 and FT030 are debug/diagnostic features. They can corrupt output or
  file descriptors, but they do not change ordinary codegen.
- Items #5-#7 matter only if Python callers share LLVM/MLIR objects across
  threads. Current compile paths do not.
