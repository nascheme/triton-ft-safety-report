# `self.configs_timings` clobbered across concurrent tuning keys

- **Status:** Open
- **Patch:** `cache-toctou.patch` (`benchmark()` returns `timings`;
  `check_disk_cache` uses the returned value for the disk write instead
  of reading `self.configs_timings`; per-key sync also prevents
  cross-key overwrites on the attribute)
- **Severity:** Significant
- **Component:** `python/triton/runtime/autotuner.py`
- **Tier:** 2

- **Shared state:** `self.configs_timings` — a `dict` mapping `Config` objects
  to benchmark timings, stored on the shared `Autotuner` instance.
- **Writer(s):** `benchmark()` inner function at line 235
  (`self.configs_timings = timings`); `check_disk_cache()` at line 199 when
  loading from disk (`self.configs_timings = timings`).
- **Reader(s):** `check_disk_cache()` at line 208 reads
  `self.configs_timings.items()` for disk-cache serialization.
- **Race scenario:** Thread A benchmarks key K1, sets
  `self.configs_timings` to K1's timings. Thread B benchmarks key K2,
  overwrites `self.configs_timings` with K2's timings. Thread A's
  `check_disk_cache` then serializes K2's timings to disk under K1's cache
  key. Future cold starts loading K1's disk cache entry get wrong timings and
  potentially select a suboptimal config.
- **Consequences:** Corrupted disk cache entries — wrong timings persisted
  under wrong keys. A poisoned entry persists across process restarts and
  silently selects the wrong autotuning config for the affected kernel
  specialization.
- **Suggested fix:** Make `configs_timings` a local variable returned from
  `benchmark()` and passed into the disk-cache write path, rather than storing
  it on `self`. Same direction as the `nargs` fix (issue #2).
