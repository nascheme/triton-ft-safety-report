# python/src/main.cc Free-Threading Issues

Scope: the top-level pybind11 module entry point for `triton._C.libtriton`.

`main.cc` mostly wires submodules together under CPython's import lock. The
substantive shared state lives in the callee files and is tracked in their
component summaries.

## Issues

| # | Severity | Tier | Component | Issue |
|---|----------|------|-----------|-------|
| 1 | LOW | 1 | `init_triton_stacktrace_hook` / `llvm::sys::AddSignalHandler` | Process-global signal handler registration sampled from `TRITON_ENABLE_PYTHON_STACKTRACE` |
| 2 | Not worth reporting | - | `PYBIND11_MODULE(libtriton, m)` body | Submodule wiring runs once under the import lock |

## Notes

- No separate issue file is warranted here.
- Env-var helpers, specialization, IR, LLVM, passes, interpreter, GSan,
  linear-layout, Gluon, and backend submodules are covered by their own
  component summaries.
