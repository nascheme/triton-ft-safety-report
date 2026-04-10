# `check_disk_cache` compound race on shared instance state and disk

- **Status:** Open
- **Severity:** Significant
- **Component:** `python/triton/runtime/autotuner.py`

- **Shared state:** `self.cache[tuning_key]` (line 198),
  `self.configs_timings` (line 199/208), and the on-disk cache file.
- **Writer(s):** `check_disk_cache` writes `self.cache` and
  `self.configs_timings` on a disk-cache hit (lines 198-199). On a disk-cache
  miss it calls `bench_fn()` (which writes both via `benchmark()`) then
  serializes `self.configs_timings` to disk (lines 203-209).
- **Reader(s):** The disk-write path at line 208 reads
  `self.configs_timings.items()`.
- **Race scenario:** Two threads with the same tuning key both pass the
  `key not in self.cache` check in `run()` (issue #1) and both enter
  `check_disk_cache`. Both call `cache.get_file()` and both miss. Both run
  `bench_fn()`, both write `self.configs_timings` (issue #3), and both call
  `cache.put()`. The `cache.put()` at line 203 reads `self.configs_timings`
  which may already have been overwritten by the other thread — serializing
  the wrong timings under this key's disk entry. With different keys, the
  cross-contamination from issue #3 flows directly into a persistent disk
  write.
- **Consequences:** Corrupted disk cache entries that persist across process
  restarts. A poisoned entry causes suboptimal or wrong config selection on
  every subsequent cold start for the affected kernel specialization.
- **Suggested fix:** Make `configs_timings` local (same as issue #3 fix) and
  pass it into the disk-write path. Add per-key synchronization so only one
  thread populates a given tuning key (same as issue #1 fix). Both fixes are
  needed to fully resolve this compound race.
