# runtime/autotuner.py Free-Threading Issues

Scope: `python/triton/runtime/autotuner.py`.

The core problem is one shared `Autotuner` instance using instance attributes
as per-call scratch state. Concurrent `run()` calls can benchmark the same key
twice and clobber hook/pruning/disk-cache context.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| FT002 | HIGH | Autotuner.cache | 2 | [`run()` cache-miss TOCTOU enables duplicate benchmarking and cross-thread hook-state contamination](autotuner/cache-toctou.md) |
| FT006 | MED | Autotuner.nargs | 2 | [`self.nargs` clobbered by concurrent `run()` calls, corrupting hook/pruning context](autotuner/nargs-clobber.md) |
| FT004 | MED | Autotuner.configs_timings | 2 | [`self.configs_timings` clobbered across concurrent tuning keys](autotuner/configs-timings-clobber.md) |
| FT007 | MED | Autotuner.restore_copies | 2 | [`self.restore_copies` pre/post hook race restores wrong snapshot or raises](autotuner/restore-copies-race.md) |
| FT001 | LOW | Autotuner.best_config | 2 | [`self.best_config` and `self.bench_time` stale across concurrent calls](autotuner/best-config-stale.md) |
| FT005 | LOW | Autotuner.do_bench | 2 | [`@cached_property` `do_bench` duplicate execution on shared instance](autotuner/do-bench-cached-property.md) |
| FT003 | MED | check_disk_cache | 2 | [`check_disk_cache` compound race on shared instance state and disk](autotuner/check-disk-cache-race.md) |

## Notes

- FT002 is the umbrella fix. It should move benchmark scratch state out of
  shared instance attributes and coalesce cache population per tuning key.
- FT006, FT004, FT007, and FT003 are concrete ways the same shared scratch
  state leaks across threads.
- FT001 and FT005 are lower priority: stale diagnostics and duplicate
  `cached_property` work.
