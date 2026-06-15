# runtime/autotuner.py Free-Threading Issues

Scope: `python/triton/runtime/autotuner.py`.

The core problem is one shared `Autotuner` instance using instance attributes
as per-call scratch state. Concurrent `run()` calls can benchmark the same key
twice and clobber hook/pruning/disk-cache context.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT002 | Critical Blocker | Autotuner.cache | [`run()` cache-miss TOCTOU enables duplicate benchmarking and cross-thread hook-state contamination](autotuner/cache-toctou.md) |
| FT006 | Blocker | Autotuner.nargs | [`self.nargs` clobbered by concurrent `run()` calls, corrupting hook/pruning context](autotuner/nargs-clobber.md) |
| FT004 | Blocker | Autotuner.configs_timings | [`self.configs_timings` clobbered across concurrent tuning keys](autotuner/configs-timings-clobber.md) |
| FT007 | Blocker | Autotuner.restore_copies | [`self.restore_copies` pre/post hook race restores wrong snapshot or raises](autotuner/restore-copies-race.md) |
| FT001 | Low | Autotuner.best_config | Concurrent calls can leave `self.best_config` and `self.bench_time` showing another thread's most recent tuning result. This is stale diagnostic/state exposure, not the root cache race. |
| FT005 | Low | Autotuner.do_bench | The shared `@cached_property` can run duplicate benchmark-helper setup on concurrent first use. The result should converge, so the impact is wasted work rather than incorrect tuning. |
| FT003 | Blocker | check_disk_cache | [`check_disk_cache` compound race on shared instance state and disk](autotuner/check-disk-cache-race.md) |

## Notes

- FT002 is the umbrella fix. It should move benchmark scratch state out of
  shared instance attributes and coalesce cache population per tuning key.
- FT006, FT004, FT007, and FT003 are concrete ways the same shared scratch
  state leaks across threads.
- FT001 and FT005 are lower priority: stale diagnostics and duplicate
  `cached_property` work.
