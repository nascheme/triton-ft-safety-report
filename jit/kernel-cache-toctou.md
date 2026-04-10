# `kernel_cache` / `kernel_key_cache` TOCTOU causes duplicate compilation

- **Status:** Open
- **Severity:** Significant
- **Component:** `runtime/jit.py`

- **Shared state:** `kernel_cache` and `kernel_key_cache` -- plain dicts inside
  the per-device `device_caches` tuple, shared across all threads calling the
  same `JITFunction` on the same device
- **Writer(s):** `_do_compile()` at lines 877, 882, 885;
  `compute_cache_key()` at line 594
- **Reader(s):** `run()` at line 732; `compute_cache_key()` at line 576
- **Race scenario:** Thread A calls `run()`, `kernel_cache.get(K)` returns
  `None`, enters `_do_compile`. Thread B does the same before A finishes
  compiling. Both compile the same kernel from scratch and both write
  `kernel_cache[K]`; last writer wins. On the async path, a further ordering
  hazard exists: `finalize_compile` (line 877) can write the real kernel
  before the placeholder write at line 882 executes, leaving a stale
  `FutureKernel` in the cache (see triage notes in README.md).
- **Suggested fix:** Guard the check-then-compile sequence with a per-device
  lock using double-checked locking. The `compute_cache_key` TOCTOU is benign
  duplicate work and does not need a separate fix.
