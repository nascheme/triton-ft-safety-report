# `kernel_cache` / `kernel_key_cache` TOCTOU causes duplicate compilation

- **Status:** Unreported
- **Severity:** Significant
- **Component:** runtime/jit.py
- **Tier:** 2
- **Source reports:** [README.md](README.md) (issues #4, #8)

## Description

Each device entry in `JITFunction.device_caches` contains two plain dicts:
`kernel_cache` (maps specialization keys to compiled kernels) and
`kernel_key_cache` (maps raw specialization tuples to string cache keys). Both
are created as empty dicts in `create_binder` (line 682) and populated lazily.

In free-threaded CPython, `dict.get()` and `dict.__setitem__()` are internally
thread-safe — concurrent operations will not corrupt the dict structure. The
issue is the **check-then-act (TOCTOU)** pattern that spans multiple dict
operations:

```python
# run(), lines 731-739
key = compute_cache_key(kernel_key_cache, specialization, options)
kernel = kernel_cache.get(key, None)       # CHECK

if kernel is None:                         # ACT on stale check
    ...
    kernel = self._do_compile(key, ...)    # expensive compilation
```

Two threads with the same specialization key can both observe `None` and both
enter the compile path.

## TOCTOU #1: `run()` → `_do_compile()` — duplicate kernel compilation

### Shared state

`kernel_cache`: a plain `dict` inside the per-device `device_caches` tuple,
shared across all threads calling the same `JITFunction` on the same device.

### Writer(s)

- `_do_compile()` sync path, line 885: `kernel_cache[key] = kernel`
- `_do_compile()` async path, line 882: `kernel_cache[key] = kernel`
  (FutureKernel placeholder)
- `finalize_compile` callback, line 877: `kernel_cache[key] = kernel`
  (real compiled kernel, replacing the FutureKernel)

### Reader(s)

- `run()` line 732: `kernel = kernel_cache.get(key, None)`

### Race scenario (sync path)

1. Thread A calls `run()` with args that produce specialization key K.
   Line 732: `kernel_cache.get(K)` returns `None`.
2. Thread B calls `run()` with the same args on the same device.
   Line 732: `kernel_cache.get(K)` also returns `None` (Thread A has not
   finished compiling yet).
3. Both threads enter `_do_compile`. Both compile the same kernel from
   scratch — an expensive operation involving MLIR/LLVM lowering and GPU code
   generation.
4. Both write `kernel_cache[K] = kernel` (lines 885). Last writer wins. The
   other compiled kernel is discarded.

**Consequence:** Duplicate GPU kernel compilation. Triton compilation can take
seconds to tens of seconds. Under high concurrency (e.g., multiple threads
launching the same kernel with the same input shapes on first use), this
wastes significant CPU and potentially GPU resources.

### Race scenario (async path)

1. Thread A (in an `AsyncCompileMode` context) hits the cache miss, enters
   `_do_compile` async path. `async_mode.submit()` returns a `FutureKernel`.
   Line 882: `kernel_cache[K] = FutureKernel_A`.
2. Thread B calls `run()` just before Thread A's write at line 882.
   Line 732: `kernel_cache.get(K)` returns `None`.
3. Thread B enters `_do_compile`. If Thread B is in a sync context, it
   compiles synchronously — duplicating the work the executor is already
   doing for Thread A. If Thread B is also in an async context, it submits
   another async compile.
4. Thread B writes `kernel_cache[K] = kernel_B` (line 885), overwriting
   Thread A's `FutureKernel_A`.
5. Thread A's `finalize_compile` callback later fires (line 877) and writes
   `kernel_cache[K] = real_kernel_A`, overwriting Thread B's entry.

**Consequence:** Same duplicate compilation, plus the cache entry flip-flops
between different objects. Any thread that captured a reference to the
overwritten `FutureKernel_A` from a previous read still works (it holds its
own future), but the kernel_cache contents are unpredictable.

## TOCTOU #1b: async path — placeholder overwrites a finalized entry

The same unsynchronized `kernel_cache` access on the async path creates a
second ordering hazard, distinct from the duplicate-compile race above.

After `async_mode.submit` returns, the `FutureKernel` is already stored in
`async_mode.future_kernels[cache_key]` and its underlying executor future is
already running. Before Thread A reaches the placeholder write at
jit.py:882, another participant can observe the future and call
`FutureKernel.result()` → `finalize_compile`:

1. Thread A (async context): `async_mode.submit(...)` returns
   `FutureKernel_A`. The executor is already running the compile.
2. Another thread executing `AsyncCompileMode.__exit__` concurrently — or
   any thread that observes `FutureKernel_A` via the `async_mode` mapping —
   picks up the future via `as_completed(self.raw_futures)`, calls
   `future_kernel.result()`, and runs `finalize_compile(kernel)`. That
   executes `kernel_cache[key] = real_kernel` **before** Thread A's line
   882 runs.
3. Thread A then executes line 882: `kernel_cache[key] = FutureKernel_A`,
   overwriting the real kernel with the placeholder.

**Consequence:** `kernel_cache[key]` ends up holding a placeholder
`FutureKernel` whose `finalize_compile` has already fired. Subsequent
`run()` calls that read this entry get the `FutureKernel`; calling any
attribute on it goes through `FutureKernel.__getattr__` → `result()`, which
observes `self.kernel is not None` and returns the real kernel.
Functionally the path still works, but the cache is left in an
inconsistent-looking state (a placeholder wrapping an already-finalized
kernel), and diagnostics that inspect `kernel_cache` directly will see the
wrong object type.

This is less severe than the duplicate-compile race but has the same root
cause and the same fix: the per-device lock must cover both the
placeholder write at jit.py:882 and the `finalize_compile` write at
jit.py:877, so the two writes cannot interleave. See
[async-compile-races.md](async-compile-races.md) for the separate
`AsyncCompileMode`/`FutureKernel` locking fix, which is necessary but not
sufficient — the ordering hazard here is inside the `_do_compile` caller
and is only closed by serializing the `kernel_cache` writes themselves.

## TOCTOU #2: `compute_cache_key()` — duplicate key computation

### Shared state

`kernel_key_cache`: a plain `dict` inside the per-device `device_caches` tuple.

### Writer(s)

- `compute_cache_key()` line 594: `kernel_key_cache[key] = cache_key`

### Reader(s)

- `compute_cache_key()` line 576: `kernel_key_cache.get(key, None)`

### Race scenario

1. Thread A calls `compute_cache_key` with specialization S. Line 576:
   `kernel_key_cache.get(S)` returns `None`.
2. Thread B does the same, also gets `None`.
3. Both compute the string cache key (lines 581-593) and both write it
   (line 594).

**Consequence:** Duplicate computation of the cache key string. The result is
deterministic so both threads write the same value. This is **minor** — the
computation is cheap compared to kernel compilation. Noted here because it
shares the same dict and the same TOCTOU pattern, and is listed as issue #8
in the README.

## Suggested fix

Guard the check-then-compile sequence with a per-key or per-cache lock.
A simple approach is a per-device lock:

```python
def create_binder(self):
    ...
    lock = threading.Lock()
    return {}, {}, target, backend, binder, lock

def run(self, *args, grid, warmup, **kwargs):
    ...
    kernel_cache, kernel_key_cache, target, backend, binder, lock = self.device_caches[device]
    ...
    key = compute_cache_key(kernel_key_cache, specialization, options)
    kernel = kernel_cache.get(key, None)
    if kernel is None:
        with lock:
            # Double-check after acquiring the lock.
            kernel = kernel_cache.get(key, None)
            if kernel is None:
                ...
                kernel = self._do_compile(key, ...)
```

A more granular approach would use per-key locks or a concurrent dict wrapper,
but the per-device lock is simpler and sufficient since the critical section
is the compile decision, not the dict access.

The `compute_cache_key` TOCTOU (#2) does not need a separate fix — it is
benign duplicate work. If the per-device lock from the kernel_cache fix is
held during the full `run()` compile path, it would also serialize
`compute_cache_key` calls for the same device.
