# `JITCallable.used_global_vals` unsynchronized read skips global-changed safety check

- **Issue-Id:** FT026
- **Status:** Open
- **Rank:** Blocker
- **Component:** `runtime/jit.py`
- **Patch:** [`used-global-vals-unsynchronized-read.patch`](used-global-vals-unsynchronized-read.patch)
- **Reproducer:** [`used-global-vals-unsynchronized-read-reproducer.py`](used-global-vals-unsynchronized-read-reproducer.py)

- **Shared state:** `JITCallable.used_global_vals` -- a dict recording captured
  global variables and their values at first compilation. Initialized to `{}`
  in `__init__`, rebound to a populated dict inside `cache_key` under
  `self._hash_lock`.
- **Writer(s):** `cache_key` writes exactly once, under `_hash_lock`
- **Reader(s):** `run()` iterates the dict with **no lock held** to check
  whether captured globals have changed
- **Race scenario:** Thread A compiles and populates `used_global_vals` under
  `_hash_lock`, then publishes the kernel to `kernel_cache`. Thread B calls
  `run()`, finds the kernel in `kernel_cache` (cache hit), but reads
  `used_global_vals` and still observes the initial empty dict -- the
  `kernel_cache` write and the `used_global_vals` write are under unrelated
  critical sections with no ordering guarantee. Thread B iterates zero entries
  and silently skips the global-changed safety check.
- **Suggested fix:** Call `self.cache_key` once at the top of `run()` to
  ensure `used_global_vals` is populated and its write is synchronized via
  `_hash_lock`. After the first call this is a cheap uncontended lock acquire.
- **Reproducer note:** The reproducer is a deterministic state-space test, not a
  probabilistic stress test. It manually creates the cache-hit state a racing
  thread can observe: `kernel_cache` has a hit while `used_global_vals` still
  appears as its initial empty dict.

## Validation

Validated locally against a source-built editable Triton checkout at
`ac580018f1` with Python 3.14.5:

```text
FT026 reproduced: cache-hit run skipped the global-changed check
  used_global_vals was empty even though CAPTURED_GLOBAL changed
forcing cache_key first raises as expected
  Global variable CAPTURED_GLOBAL has changed since we compiled this kernel, from 1 to 2
```
