# runtime/_async_compile.py Free-Threading Issues

Issues in `python/triton/runtime/_async_compile.py` affecting free-threaded
Python 3.14t.

File in scope:

- `runtime/_async_compile.py` — `AsyncCompileMode` (context-manager wrapper
  around a `concurrent.futures.Executor`), `FutureKernel` (lazy wrapper
  around a `Future` that runs a `finalize_compile` callback on first
  `.result()`), and the module-level `active_mode: ContextVar`.

This module is small (~70 lines) but sits on the async-compile hot path in
`runtime/jit.py:_do_compile`. Two threads can reach the same
`AsyncCompileMode` and the same `FutureKernel` concurrently through two
distinct call paths:

- the `AsyncCompileMode.__exit__` `as_completed` loop (the submitter's
  thread), and
- `run()` -> `kernel_cache.get(key)` -> `FutureKernel.__getattr__` ->
  `result()` (any thread that hits the placeholder entry that
  `jit.py:897` wrote into `kernel_cache`).

Most of the hazards in this file were originally triaged as part of the
`jit.py` pass and written up under `jit/async-compile-races.md`. This
component issues list consolidates them and adds a few file-local observations
(executor list mutation, `__getattr__` recursion, `__enter__`/`__exit__`
mismatch) that did not need to live in the jit writeup.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | Significant | AsyncCompileMode.submit | 2 | [`submit()` get-then-set TOCTOU on `future_kernels` causes duplicate executor submissions](submit-toctou.md) |
| 2 | Significant | FutureKernel.result | 2 | [`result()` non-atomic first-use causes duplicate `finalize_compile` / duplicate `jit_post_compile_hook`](future-kernel-result-race.md) |
| 3 | Minor | AsyncCompileMode.raw_futures | 2 | [`raw_futures.append` during `as_completed` iteration in `__exit__`](raw-futures-list-mutation.md) |
| 4 | Minor | FutureKernel.__getattr__ | 2 | [`__getattr__` re-enters `result()` from every attribute access, widening the race window](getattr-result-reentry.md) |

All four issues are realistically reachable only in Tier 2 scenarios where
two threads share either the same `AsyncCompileMode` instance or the same
`FutureKernel` stashed in a shared `kernel_cache`. Issues #1 and #2 are
the ones to care about; #3 and #4 are minor follow-ups.

> **Cross-reference.** Issues #1 and #2 are already written up in
> [`jit/async-compile-races.md`](../jit/async-compile-races.md); the file
> there is the canonical fix target (the patch `async-compile-races.patch`
> lives alongside it). The writeups in this component exist so that the
> `_async_compile.py` pass stands on its own and so that the deep-dive
> agent can verify them against the file directly.

## Triage notes

### 1. `AsyncCompileMode.submit` — get-then-set TOCTOU on `future_kernels`

- **Shared state:** `AsyncCompileMode.future_kernels` dict (line 44) and
  `raw_futures` list (line 43).
- **Code (_async_compile.py:45-55):**
  ```python
  def submit(self, key, compile_fn, finalize_fn):
      future = self.future_kernels.get(key)
      if future is not None:
          return future

      future = self.executor.submit(compile_fn)
      future._key = key
      self.raw_futures.append(future)
      future_kernel = FutureKernel(finalize_fn, future)
      self.future_kernels[key] = future_kernel
      return future_kernel
  ```
- **Reader:** this function, and implicitly `__exit__` via `future._key`.
- **Writer:** this function; also the sole writer of `future._key` (a
  custom attribute on the `Future` object).
- **Race:** Two threads call `submit(key, …)` with the same `key` from
  `jit.py:896`. Both see `future_kernels.get(key) -> None`, both call
  `executor.submit(compile_fn)`, both append to `raw_futures`, both
  write `future_kernels[key]`. The dict write itself is internally
  thread-safe on free-threaded CPython (last write wins), but the
  function's memoization contract is broken — two independent compile
  jobs are enqueued on the executor for the same cache key, and both
  `FutureKernel`s eventually run their `finalize_compile` callbacks.
  That means `jit_post_compile_hook` fires twice per key and
  `kernel_cache[key]` is written twice with two different
  `CompiledKernel` objects (issue #4 in `jit/issues.md`).
- **Tier:** 2.
- **Suggested fix:** add an `AsyncCompileMode._lock = threading.Lock()`
  in `__init__` and hold it across the get / submit / append / set
  sequence. The critical section is short; the executor call itself is
  non-blocking.

### 2. `FutureKernel.result` — non-atomic first-use

- **Shared state:** `FutureKernel.kernel` attribute (set in `__init__`
  at line 13 to `None`, overwritten in `result()` at line 28).
- **Code (_async_compile.py:16-29):**
  ```python
  def result(self, ignore_errors: bool = False):
      if self.kernel is not None:
          return self.kernel
      try:
          kernel = self.future.result()
      except Exception:
          if ignore_errors:
              return
          else:
              raise
      self.finalize_compile(kernel)
      self.kernel = kernel
      return kernel
  ```
- **Reader/Writer:** every call site: `AsyncCompileMode.__exit__` for
  the submitter, and `FutureKernel.__getattr__` for any thread that
  calls an attribute on the `FutureKernel` placeholder stored in
  `kernel_cache` (`jit.py:897`).
- **Race:** Two threads on the same `FutureKernel` instance both observe
  `self.kernel is None`, both call `self.future.result()` (which is
  internally thread-safe and returns the same `CompiledKernel`), both
  call `self.finalize_compile(kernel)`. In the `jit.py` async path
  `finalize_compile` is a closure that writes `kernel_cache[key]` and
  fires `jit_post_compile_hook` (jit.py:891-894) — both fire twice.
  The duplicate `kernel_cache[key]` write is benign (same kernel), but
  the duplicate hook invocation is user-observable.
- **Ordering hazard:** separately, `result()` can complete
  `finalize_compile` **before** the submitter's
  `kernel_cache[key] = kernel` placeholder write at `jit.py:897`
  executes. The placeholder then overwrites the finalized real kernel
  with the `FutureKernel` wrapper. Functionally correct (the wrapper's
  `__getattr__` proxies via `result()` which returns the memoized
  `self.kernel`), but the cache entry type is wrong. This is already
  tracked in `jit/issues.md` under #4 and #9.
- **Tier:** 2.
- **Suggested fix:** add a `threading.Lock` to `FutureKernel` and
  double-check `self.kernel` under the lock before calling
  `self.finalize_compile`. Make `finalize_compile` effectively
  idempotent via the memoization.

### 3. `AsyncCompileMode.raw_futures` — list mutation during `as_completed`

- **Shared state:** `self.raw_futures: list` (line 42).
- **Code (_async_compile.py:63-67):**
  ```python
  def __exit__(self, exc_type, exc_value, traceback):
      active_mode.set(None)
      for future in as_completed(self.raw_futures):
          self.future_kernels[future._key].result(self.ignore_errors)
  ```
- **Race:** `concurrent.futures.as_completed` takes a snapshot of the
  iterable into its own set at call time, so the concurrent `.append`
  in `submit` does not corrupt the iteration. However, any `submit()`
  call that arrives after `__exit__` started will not be awaited by
  that `as_completed` loop, and the submitter's `active_mode` is
  already cleared in the current thread's context. The window exists
  only if some other thread still holds a reference to this
  `AsyncCompileMode` and continues to call `submit`.
- **Severity:** Minor. The typical user pattern is one thread entering
  and exiting the `with AsyncCompileMode(...)` block, so this race
  requires a somewhat pathological sharing of the mode across threads.
- **Follow-up for the deep-dive agent:** verify that
  `concurrent.futures.as_completed` really does snapshot, and audit
  whether `active_mode.set(None)` in `__exit__` can leak the `None`
  into a caller's outer `ContextVar` state (see also note on the
  `__enter__`/`__exit__` mismatch below).

### 4. `FutureKernel.__getattr__` re-enters `result()`

- **Code (_async_compile.py:31-34):**
  ```python
  def __getattr__(self, name):
      return getattr(self.result(), name)
  ```
- **Concern:** Every missing attribute lookup on a `FutureKernel`
  — including innocuous ones like `.name`, `.module`, `.metadata`, or
  any `hasattr` check — routes through `result()`. Under free-threading
  that means the issue #2 race is exposed to every casual attribute
  access from every thread that touches the `FutureKernel` placeholder
  that `jit.py:897` stuffed into `kernel_cache`. It also means the
  `finalize_compile` callback (which writes `kernel_cache[key]` and
  fires a user hook) can be triggered as a side effect of apparently
  read-only code elsewhere in the runtime.
- **Severity:** Minor on its own — it is the same race as #2, but it
  widens the set of triggering code paths that a deep-dive agent must
  keep in mind when reasoning about when `finalize_compile` actually
  fires. Fixing #2 fixes this.
- **Follow-up for the deep-dive agent:** enumerate the callers that
  touch a `FutureKernel` as if it were a `CompiledKernel` (e.g., in
  `jit.py:run`, `autotuner.py`, tests) and confirm that none of them
  rely on "read-only" attribute access being side-effect free.

## Not worth reporting (verify during deep dive)

### `active_mode: ContextVar[Optional[AsyncCompileMode]]`

- **Code (_async_compile.py:6):**
  ```python
  active_mode: ContextVar[Optional[AsyncCompileMode]] = ContextVar(
      "async_compile_active_mode", default=None)
  ```
- **Why carved out by CLAUDE.md:** `ContextVar`s are per-context, not
  process-global; the CLAUDE.md "Low-value patterns" section
  explicitly excludes them unless they leak. Each thread starts with
  its own context, so `active_mode.get()` in thread A does not see
  `active_mode.set(self)` in thread B.
- **Worth verifying anyway:**
  1. `__enter__` / `__exit__` use `.set(value)` rather than capturing
     a `Token` and calling `.reset(token)`. If a caller ever nested
     two `AsyncCompileMode` blocks (even by accident via library
     code), the inner `__exit__` resets `active_mode` to `None`
     instead of restoring the outer mode. The `__enter__` guard at
     line 58 makes nested re-entry raise, so this is only a risk if
     the guard is bypassed or if a child context propagates. Note
     this explicitly for the deep-dive agent.
  2. `asyncio`/`concurrent.futures` executor threads that inherit a
     copied context could observe a stale `active_mode` value. The
     executor passed into `AsyncCompileMode` runs `compile_fn` in a
     worker thread; `compile_fn` is a plain closure that does not
     consult `active_mode`, so this is not a current hazard. Confirm
     during deep dive.

### `AsyncCompileMode.future_kernels` / `raw_futures` writes considered in isolation

- Individual `dict.__setitem__` and `list.append` calls are internally
  thread-safe on free-threaded CPython (see
  `PYTHON_THREADSAFETY.md`). The concerns above are about the
  compound sequences, not the individual operations.

### `__init__` field writes

- `AsyncCompileMode.__init__` (lines 39-44) and `FutureKernel.__init__`
  (lines 11-14) only run during object construction. Until the
  reference escapes to another thread (via `active_mode.set(self)` or
  via `kernel_cache[key] = future_kernel`), no other thread can see
  the instance. Write-once construction.

### `future._key = key` custom attribute assignment

- Set once in `submit` at line 51, read once in `__exit__` at line 67.
  No concurrent mutation. `concurrent.futures.Future` does not use
  `__slots__`, so the assignment succeeds. Benign.
