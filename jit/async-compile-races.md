# `AsyncCompileMode` / `FutureKernel` are not thread-safe

- **Status:** Open
- **Patch:** `async-compile-races.patch` (lock around `AsyncCompileMode.submit` get/submit/store, and `FutureKernel._lock` making `finalize_compile` idempotent)
- **Severity:** Significant
- **Component:** `runtime/_async_compile.py`
- **Tier:** 2

Two threads can reach the same `FutureKernel` concurrently: one via the
`AsyncCompileMode.__exit__` `as_completed` loop, another via
`kernel_cache.get()` -> `FutureKernel.__getattr__` -> `result()` in `run()`.
See triage notes in issues.md for how threads reach these paths.

## Bug 1: `AsyncCompileMode.submit` TOCTOU

- **Shared state:** `AsyncCompileMode.future_kernels` dict and `raw_futures`
  list
- **Race scenario:** Two threads call `submit` with the same `cache_key`.
  Both see `future_kernels.get(key)` -> `None`, both call
  `executor.submit(compile_fn)` -- two independent compile jobs are enqueued
  for the same source, defeating the `future_kernels` memoization. Duplicate
  `jit_post_compile_hook` invocations follow.

## Bug 2: `FutureKernel.result` duplicate `finalize_compile`

- **Shared state:** `FutureKernel.kernel` attribute
- **Race scenario:** Two threads call `result()` on the same `FutureKernel`.
  Both observe `self.kernel is None`, both call `self.future.result()` (which
  is internally thread-safe and returns the same kernel), both call
  `self.finalize_compile(kernel)`. User hooks fire twice for one compile.

## Suggested fix

Add a lock to `AsyncCompileMode.submit` covering the get-submit-store
sequence, and a lock to `FutureKernel.result` making `finalize_compile`
idempotent. Both changes are local to `_async_compile.py`.
