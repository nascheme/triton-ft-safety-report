# `AsyncCompileMode` / `FutureKernel` are not thread-safe

- **Status:** Unreported
- **Severity:** Significant
- **Component:** `runtime/_async_compile.py` (with callers in `runtime/jit.py`)
- **Tier:** 2 (multiple threads calling the same kernel) and 3 (hooks mutated on
  shared mode object)
- **Source reports:** [README.md](README.md) (issue #9)

## Background

When a `JITFunction` is invoked inside an `AsyncCompileMode` context, the
compile path in `_do_compile` (jit.py:859-888) takes an async branch:

```python
async_mode = _async_compile.active_mode.get()
if async_mode is not None:
    env_vars = get_cache_invalidating_env_vars()
    cache_key = get_cache_key(src, backend, options, env_vars)

    def async_compile():
        return self.compile(src, target=target, options=options.__dict__, _env_vars=env_vars)

    def finalize_compile(kernel):
        kernel_cache[key] = kernel
        self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, target, device, constexprs,
                        options, [attrs], warmup)

    kernel = async_mode.submit(cache_key, async_compile, finalize_compile)
    kernel_cache[key] = kernel
```

`async_mode.submit` returns a `FutureKernel` placeholder wrapping an
`Executor.submit(...)` future. The placeholder is stored into `kernel_cache`
under the JIT key, and the real compiled kernel replaces it later when
`FutureKernel.result()` is called and invokes `finalize_compile`.

`AsyncCompileMode` memoizes by `cache_key` so that the same source compiled
twice inside the same async batch reuses one executor future. `FutureKernel`
memoizes the compile result so that repeated `result()` calls are cheap.

Neither memoization is thread-safe. Both bugs are fixed together by adding
locking to `AsyncCompileMode.submit` and `FutureKernel.result`.

## Why threads can reach here concurrently

`AsyncCompileMode` is held in a `ContextVar` (`_async_compile.active_mode`),
so by default a child thread spawned from a thread inside the async context
does **not** see the same mode (`ContextVar`s are per-context). However:

1. A thread can explicitly capture and reuse the mode object across threads
   (e.g., pass it to `concurrent.futures.ThreadPoolExecutor`, or set it in
   another thread via `active_mode.set(mode)` inside
   `contextvars.copy_context`).
2. Even without sharing the `AsyncCompileMode` instance, the `kernel_cache`
   dict on the per-device `device_caches` entry is shared across all threads
   calling the same `JITFunction` on the same device. A non-async thread can
   observe the placeholder `FutureKernel` and call methods on it, triggering
   `FutureKernel.result()` → `finalize_compile` on a thread that is not the
   one that submitted the compile.

Case (2) is the most realistic free-threading scenario: one thread runs a
batch warmup under `AsyncCompileMode`, another thread launches a kernel
concurrently (e.g., an unrelated model). The launching thread's `run()`
hits `kernel_cache.get(key)` and gets the `FutureKernel`, then calls
`kernel.launch_metadata(...)` which goes through `__getattr__` → `result()`
→ `finalize_compile`. Meanwhile the warmup thread's `__exit__` also calls
`result()` on the same `FutureKernel` in its `as_completed` loop. Both
threads run `finalize_compile` concurrently.

## Bug 1: `AsyncCompileMode.submit` check-then-insert TOCTOU

### Shared state

`AsyncCompileMode.future_kernels: dict[cache_key, FutureKernel]` and
`AsyncCompileMode.raw_futures: list[Future]` on a mode object that may be
shared across threads.

### Code

```python
# _async_compile.py:45-55
def submit(self, key, compile_fn, finalize_fn):
    future = self.future_kernels.get(key)       # CHECK
    if future is not None:
        return future

    future = self.executor.submit(compile_fn)   # expensive: enqueues compile
    future._key = key
    self.raw_futures.append(future)
    future_kernel = FutureKernel(finalize_fn, future)
    self.future_kernels[key] = future_kernel    # ACT
    return future_kernel
```

### Race

Two threads enter `submit` with the same `cache_key`:

1. Thread A: `future_kernels.get(key)` → `None`.
2. Thread B: `future_kernels.get(key)` → `None` (A has not written yet).
3. Both call `self.executor.submit(compile_fn)` — **two independent compile
   jobs are enqueued for the same source**.
4. Both append to `raw_futures` and construct distinct `FutureKernel`
   wrappers.
5. Both write `future_kernels[key] = future_kernel`; last writer wins.

### Consequence

- Duplicate kernel compilation inside an async compile batch — the exact
  thing the `future_kernels` memoization is supposed to prevent.
- The losing `FutureKernel` is still referenced by whichever thread's
  `_do_compile` received it, and by the `raw_futures` list. Its
  `finalize_compile` callback will still fire when its future completes
  (via `__exit__`'s `as_completed` loop), so the post-compile hook chain
  fires an extra time and `kernel_cache[key] = kernel` is written an extra
  time (same value, so benign for `kernel_cache` but not for the hook).

## Bug 2: `FutureKernel.result` non-atomic — duplicate `finalize_compile`

### Shared state

`FutureKernel.kernel` and the underlying `concurrent.futures.Future`.

### Code

```python
# _async_compile.py:16-29
def result(self, ignore_errors: bool = False):
    if self.kernel is not None:
        return self.kernel
    try:
        kernel = self.future.result()
    except Exception:
        ...
    self.finalize_compile(kernel)
    self.kernel = kernel
    return kernel
```

### Race

Two threads obtain the same `FutureKernel` (either via
`kernel_cache.get(key)` seeing the placeholder written at jit.py:882, or
via `future_kernels[cache_key]` in `__exit__`'s `as_completed` loop) and
both call `result()`:

1. Both observe `self.kernel is None`.
2. Both call `self.future.result()` — `concurrent.futures.Future.result()`
   is internally thread-safe and returns the same compiled kernel to both.
3. **Both call `self.finalize_compile(kernel)`.** For the `_do_compile`
   closure this runs:
   - `kernel_cache[key] = kernel` — idempotent, same value, safe.
   - `self._call_hook(knobs.runtime.jit_post_compile_hook, ...)` — **user
     hook called twice for the same compile**.
4. Both write `self.kernel = kernel` (same value, safe).

### Consequence

`knobs.runtime.jit_post_compile_hook` is invoked twice for a single compile.
User hooks may count compiles, log them, emit telemetry, or populate
external caches; a duplicate invocation is a real observable bug. The
`_call_hook` path also wraps each hook and records statistics, so duplicate
fires can mis-attribute timings.

## Suggested fix

Both bugs are fixed together by adding locking to `_async_compile.py`:

1. **`AsyncCompileMode.submit`** — guard the
   `future_kernels.get` / `executor.submit` / `future_kernels[key] = ...`
   sequence with a `threading.Lock` on the mode object. The executor
   submission must be inside the lock so two threads cannot both enqueue a
   compile for the same `cache_key`.

2. **`FutureKernel.result`** — make the method idempotent and thread-safe:
   guard the `self.kernel is None` check and the `finalize_compile` call
   with a lock, or use a sentinel "finalized" flag set under the lock. Only
   one call to `finalize_compile` should ever happen per `FutureKernel`.

Both changes are local to `_async_compile.py` and naturally belong in a
single PR: they close the same "the async compile path is not designed for
concurrent callers" gap, and a reviewer should see them together.

### Related: placeholder-vs-finalize ordering in `_do_compile`

There is a separate ordering hazard in `jit.py` where the placeholder write
at jit.py:882 can overwrite an already-finalized `kernel_cache` entry. That
issue is caused by the unsynchronized `kernel_cache` access on the async
path and is fixed by the per-device cache lock proposed in
[kernel-cache-toctou.md](kernel-cache-toctou.md), not by changes to
`_async_compile.py`. See that report for details.
