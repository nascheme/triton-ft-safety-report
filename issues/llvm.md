# python/src/llvm.cc Free-Threading Issues

Scope: pybind11 bindings for LLVM/MLIR translation, module optimization,
object/assembly emission, diagnostics, and LLVM process-global options.

## Issues

| # | Severity | Tier | Component | Issue |
|---|----------|------|-----------|-------|
| FT029 | LOW | OOS | `setLLVMOption` / `restoreLLVMOption` / `ScopedLLVMOption` | [Process-global `llvm::cl::opt<>` state mutated from GIL-released paths](llvm/global-cl-opt-mutation.md) |
| FT028 | LOW | 1/2 | `dumpSchedulingDAG` stderr redirection | [Process-wide `dup` / `freopen(stderr)` / `dup2` sequence races with other threads and Python I/O](llvm/dump-sched-dag-stderr-redirection.md) |
| FT030 | LOW | 1/2 | `llvm::TimePassesIsEnabled` / `llvm::TimePassesPerRun` | [Process-global timing flags used from GIL-released `translateLLVMIRToASM`](llvm/time-passes-global-flags.md) |
| 4 | LOW | 2 | `init_targets` | Unconditional write to `llvm::parallel::strategy`; concurrent writers set the same value |
| 5 | LOW latent | 2 | `to_module` | Unsafe on a shared `mlir::ModuleOp` / `LLVMContext`, but shipping paths use per-compile objects |
| 6 | LOW latent | 2 | `optimize_module` | Unsafe on a shared `llvm::Module`, but shipping paths use per-compile modules |
| 7 | LOW latent | 2 | LLVM module mutation bindings | Unsafe on a shared module/function object; caller contract today |
| 8 | LOW | 3 | `init_triton_stacktrace_hook` | Process-global signal handler registration |

## Notes

- FT029 is a pre-existing maintainer-rule violation, not a Tier 1-2 ask:
  runtime mutation of LLVM `cl::opt` is already unsupported.
- FT028 and FT030 are debug/diagnostic features. They can corrupt output or
  file descriptors, but they do not change ordinary codegen.
- Items #5-#7 matter only if Python callers share LLVM/MLIR objects across
  threads. Current compile paths do not.
- The extended lock-design discussion for FT029 lives in
  [`llvm/global-cl-opt-mutation.md`](llvm/global-cl-opt-mutation.md).
