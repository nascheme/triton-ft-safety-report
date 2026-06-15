# interpreter Free-Threading Issues

Scope: `python/triton/runtime/interpreter.py` and `python/src/interpreter.cc`.

All findings here require `TRITON_INTERPRET=1`. Interpreter mode is currently a
debug-only, single-threaded path outside the current free-threading goal, so
these are Deferred today. The top Deferred issues are still serious if
concurrent interpreter execution becomes in-scope.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT014 | Deferred | `interpreter_builder` | The module-level `interpreter_builder` stores `grid_idx` / `grid_dim`, so concurrent interpreter launches can read another thread's grid state and execute with wrong `program_id` values. Deferred because it requires `TRITON_INTERPRET=1`. |
| FT017 | Deferred | `_patch_lang` / `_LangPatchScope` | Interpreter mode monkey-patches process-global `tl.*` symbols, and restore ordering can permanently corrupt them under concurrent runs. Deferred because it requires concurrent `TRITON_INTERPRET=1` execution. |
| FT015 | Deferred | `FunctionRewriter._compile_and_exec` | The rewrite path mutates a user kernel's shared `fn.__globals__` during `exec`, so concurrent interpreter compilation can leak temporary globals across threads. Deferred because interpreter concurrency is not currently supported. |
| FT016 | Deferred | `InterpretedFunction.__call__` | `__call__` applies `_patch_lang` without restoring it, so interpreter patches can leak into later calls or non-interpreter code. Deferred because it is confined to `TRITON_INTERPRET=1`. |
| 5 | Low | `InterpretedFunction.rewritten_fn` | Class-level rewrite cache is check-then-set; duplicate AST rewrite under concurrent first use |
| 6 | Low | `_patch_lang` | Iteration over `fn.__globals__.items()` while `_compile_and_exec` mutates the same dict |
| 7 | Low | `interpreter.cc` `mem_semantic_map` | `std::map` read via non-const `operator[]`; insert-on-miss latent risk |

## Notes

- Pragmatic current fix: document interpreter mode as single-threaded or
  serialize `GridExecutor.__call__` with one process-level lock.
- A real concurrent interpreter design would need thread-local/context-local
  grid state and an alternative to process-wide `tl.*` monkey-patching.
- `atomic_op_guard` correctly serializes fallback atomics but may become a
  scalability bottleneck if interpreter parallelism is ever supported.
