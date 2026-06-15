# interpreter Free-Threading Issues

Scope: `python/triton/runtime/interpreter.py` and `python/src/interpreter.cc`.

All findings here require `TRITON_INTERPRET=1`. Interpreter mode is currently a
debug-only, single-threaded path outside the current free-threading goal, so
these are Deferred today. The top Deferred issues are still serious if
concurrent interpreter execution becomes in-scope.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT014 | Deferred | `interpreter_builder` | [Global `interpreter_builder.grid_idx` / `grid_dim` trampled by concurrent kernel runs](interpreter/builder-grid-state-race.md) |
| FT017 | Deferred | `_patch_lang` / `_LangPatchScope` | [Process-wide monkey-patching of `tl.*` with save/restore permanently corrupts `tl` under concurrent runs](interpreter/patch-lang-save-restore-race.md) |
| FT015 | Deferred | `FunctionRewriter._compile_and_exec` | [Mutation of user kernel's `fn.__globals__` during `exec`, shared across threads](interpreter/compile-and-exec-globals-mutation.md) |
| FT016 | Deferred | `InterpretedFunction.__call__` | [`__call__` path applies `_patch_lang` without ever restoring, leaking patches](interpreter/interpreted-fn-call-leaks-patches.md) |
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
