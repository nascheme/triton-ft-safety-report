# `check_disk_cache` compound race on shared instance state and disk

- **Issue-Id:** FT003
- **Status:** Open
- **Rank:** Blocker
- **Component:** `python/triton/runtime/autotuner.py`
- **Patch:** [`cache-toctou.patch`](cache-toctou.patch)

- **Shared state:** `self.cache[tuning_key]`, `self.configs_timings`,
  and the on-disk cache file.
- **Writer(s):** `check_disk_cache` writes `self.cache` and
  `self.configs_timings` on a disk-cache hit. On a disk-cache miss it calls
  `bench_fn()` (which writes both via `benchmark()`) then serializes
  `self.configs_timings` to disk.
- **Reader(s):** The disk-write path in `check_disk_cache` reads
  `self.configs_timings.items()`.
- **Race scenario:** Two threads with the same tuning key both pass the
  `key not in self.cache` check in `run()` (issue FT002) and both enter
  `check_disk_cache`. Both call `cache.get_file()` and both miss. Both run
  `bench_fn()`, both write `self.configs_timings` (issue FT004), and both call
  `cache.put()`. The `cache.put()` call reads `self.configs_timings`
  which may already have been overwritten by the other thread — serializing
  the wrong timings under this key's disk entry. With different keys, the
  cross-contamination from issue FT004 flows directly into a persistent disk
  write.
- **Consequences:** Corrupted disk cache entries that persist across process
  restarts. A poisoned entry causes suboptimal or wrong config selection on
  every subsequent cold start for the affected kernel specialization.
- **Suggested fix:** Make `configs_timings` local (same as FT004 fix) and
  pass it into the disk-write path. Add per-key synchronization so only one
  thread populates a given tuning key (same as FT002 fix). Both fixes are
  needed to fully resolve this compound race.
