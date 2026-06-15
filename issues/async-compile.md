# runtime/_async_compile.py Free-Threading Issues

Scope: `python/triton/runtime/_async_compile.py`, especially
`AsyncCompileMode`, `FutureKernel`, and the module-level `active_mode`
`ContextVar`.

This component is small. The actionable behavior is already covered by
[`jit/async-compile-races.md`](jit/async-compile-races.md), because the race is
observable through `runtime/jit.py`'s shared `kernel_cache`.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| 1 | Blocker | AsyncCompileMode.submit | Duplicate executor submissions when two threads call `submit()` with the same key |
| 2 | Blocker | FutureKernel.result | Duplicate `finalize_compile` / `jit_post_compile_hook` from non-atomic first use |
| 3 | Low | AsyncCompileMode.raw_futures | `raw_futures.append` can race the `as_completed` loop in `__exit__` |
| 4 | Low | FutureKernel.__getattr__ | Attribute access re-enters `result()`, widening the race window |

## Notes

- Issues #1 and #2 are the only material findings. Use
  [`jit/async-compile-races.md`](jit/async-compile-races.md) as the canonical
  writeup and patch target.
- Issues #3 and #4 are minor consequences of the same shared
  `AsyncCompileMode` / `FutureKernel` lifecycle.
- `active_mode` itself is a `ContextVar`, so the context binding is not shared
  across threads.
- Field writes during object construction are not reported.
