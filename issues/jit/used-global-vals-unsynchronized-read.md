# `JITCallable.used_global_vals` unsynchronized read skips global-changed safety check

- **Status:** Open
- **Severity:** Significant
- **Component:** `runtime/jit.py`
- **Tier:** 2
- **Patch:** `used-global-vals-unsynchronized-read.patch`

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
