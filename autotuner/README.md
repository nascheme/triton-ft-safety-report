# runtime/autotuner.py Free-Threading Issues

Issues in `python/triton/runtime/autotuner.py` affecting free-threaded Python 3.14t.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | SEVERE | Autotuner.cache | 2 | [`run()` cache-miss TOCTOU enables duplicate benchmarking and cross-thread hook-state contamination](cache-toctou.md) |
| 2 | Significant | Autotuner.nargs | 2 | [`self.nargs` clobbered by concurrent `run()` calls, corrupting hook/pruning context](nargs-clobber.md) |
| 3 | Significant | Autotuner.configs_timings | 2 | [`self.configs_timings` clobbered across concurrent tuning keys](configs-timings-clobber.md) |
| 4 | Significant | Autotuner.restore_copies | 2 | [`self.restore_copies` pre/post hook race restores wrong snapshot or raises](restore-copies-race.md) |
| 5 | Minor | Autotuner.best_config | 2 | [`self.best_config` and `self.bench_time` stale across concurrent calls](best-config-stale.md) |
| 6 | Minor | Autotuner.do_bench | 2 | [`@cached_property` `do_bench` duplicate execution on shared instance](do-bench-cached-property.md) |
| 7 | Significant | check_disk_cache | 2 | [`check_disk_cache` compound race on shared instance state and disk](check-disk-cache-race.md) |

## Triage notes

### 1. `self.cache` — cache-miss TOCTOU races the whole benchmark path (SEVERE)

- **Shared state:** the miss path is not just `self.cache: Dict[Tuple, Config]`
  (line 37); it also uses shared per-call scratch state on the same
  `Autotuner`, including `self.nargs`, `self.restore_copies`,
  `self.configs_timings`, `self.bench_time`, and `self.best_config`.
- **Writer:** `run()` checks `if key not in self.cache`, then on a miss enters
  `benchmark()` and later writes `self.cache[key] = ...`; `check_disk_cache()`
  can also populate the cache.
- **Reader:** `run()` uses `self.cache` as the miss guard and later reads the
  chosen config back.
- **Race scenario:** Thread A and Thread B call `run()` with the same tuning
  key. Both observe a miss and both enter benchmarking. The `dict` operations
  themselves are container-safe under free-threaded CPython, but the TOCTOU
  allows concurrent execution of a benchmark path that reuses shared scratch
  state.
- **Important clarification:** `self.nargs` does **not** directly replace the
  local `*args` passed to `self.fn.run(*args, ...)` inside `_bench()`. The
  kernel launch still uses the calling thread's local positional args. The race
  instead corrupts the named-argument context (`full_nargs`) passed to
  `config.pre_hook`, `self.pre_hook`, `self.post_hook`, `early_config_prune`,
  and `perf_model`.
- **Consequences:**
  - duplicate benchmark/compile/launch work for the same key;
  - default `restore_value` / `reset_to_zero` hooks can zero, snapshot, or
    restore the wrong tensors, or fail to restore a benchmarked tensor at all;
  - `self.restore_copies` can be overwritten/cleared cross-thread, leading to
    wrong tensor contents, `KeyError`, or PyTorch `RuntimeError` from an
    incompatible `copy_()`;
  - config pre-hooks can mutate the wrong descriptor-like objects, which is more
    serious than the default tensor-hook case because it can affect benchmark
    launch state;
  - `self.configs_timings` can be serialized under the wrong disk-cache key.
- **Fix direction:** keep benchmark scratch state local to the call and add
  per-key synchronization so only one thread populates a given tuning key.
  This matches upstream PR `triton-lang/triton#8437`, which passes `nargs`
  explicitly and uses a per-key future/event for cache population. Tier 2.

### 2. `self.nargs` — clobbered by concurrent callers (Significant)

- **Shared state:** `self.nargs` (line 213), set at the start of `run()` and
  read throughout `_bench()`, `prune_configs()`, and hook callbacks.
- **Writer:** `run()` line 213 (`self.nargs = dict(zip(self.arg_names, args))`),
  also `warmup()` line 288. Cleared at line 257 (`self.nargs = None`).
- **Reader:** `_bench()` line 143 (`self.nargs`), `prune_configs()` line 263,
  `check_disk_cache` indirectly via `benchmark()`.
- **Race scenario:** Thread A calls `run(x_ptr, 1024)`, sets `self.nargs`.
  Thread B calls `run(y_ptr, 2048)`, overwrites `self.nargs`. Thread A's
  in-progress `_bench()` or `prune_configs()` now uses Thread B's arguments.
  Importantly, this does **not** directly replace Thread A's local `*args` in
  `self.fn.run(*args, ...)`; instead it corrupts the named-argument context seen
  by hooks, `early_config_prune`, and `perf_model`. The likely outcomes are
  wrong hook targets / wrong pruning decisions / wrong descriptor mutation, and
  in some interleavings `None` reads after another thread clears the attribute.
  Tier 2.

### 3. `self.configs_timings` — unsynchronized shared write (Significant)

- **Shared state:** `self.configs_timings` — set inside `benchmark()` at
  line 235 and inside `check_disk_cache()` at line 199.
- **Writer:** `benchmark()` (called from `run()`), `check_disk_cache()`.
- **Reader:** `check_disk_cache()` line 208 reads it for disk cache
  serialization. `knobs.autotuning.print` path at line 248 may also observe it.
- **Race scenario:** Thread A finishes benchmarking and sets
  `self.configs_timings`. Thread B (with a different key) also benchmarks and
  overwrites it. Thread A's `check_disk_cache` then serializes Thread B's
  timings to disk under Thread A's cache key. Or vice versa. Tier 2.

### 4. `self.restore_copies` — pre/post hook race (Significant)

- **Shared state:** `self.restore_copies` — written by `_pre_hook` closure
  (line 63) and read/cleared by `_post_hook` closure (lines 73-75).
- **Writer:** `_pre_hook` (line 63, inside `_bench` via `kernel_call`).
- **Reader:** `_post_hook` (line 73).
- **Race scenario:** Thread A's `_pre_hook` saves tensor copies into
  `self.restore_copies`. Thread B's `_pre_hook` overwrites them with its own
  tensor copies. Thread A's `_post_hook` then restores Thread B's saved
  values into Thread A's tensor, or another interleaving clears the dict before
  the matching restore. The default outcomes are silent wrong tensor contents,
  missed restore, `KeyError`, or PyTorch `RuntimeError` from an incompatible
  cross-thread `copy_()` rather than a direct native memory-safety crash.
  Tier 2.

### 5. `self.best_config` and `self.bench_time` — per-call writes (Significant)

- **Shared state:** `self.best_config` (line 245), `self.bench_time` (line 231).
- **Writer:** `run()` writes `self.best_config` at line 245, `benchmark()`
  writes `self.bench_time` at line 231.
- **Reader:** `run()` reads `self.best_config` at line 247 for printing.
  `self.bench_time` at line 248.
- **Race scenario:** Thread A sets `self.best_config` for key K1, Thread B
  then sets it for key K2. The print statement at line 247 in Thread A may
  print Thread B's config. The practical impact is limited to incorrect
  diagnostic output, but `self.bench_time` can also be stale. Tier 2.
- **Severity downgrade:** Originally listed as Significant. On review,
  `self.best_config` has no readers outside the diagnostic print (confirmed
  by grep across `triton/python/triton/`). The actual kernel launch uses the
  local `config` variable at line 242, not `self.best_config`. Downgraded to
  Minor — wrong diagnostic output only.

### 6. `do_bench` `@cached_property` — duplicate execution (Minor)

- **Shared state:** `self.do_bench` backed by `@cached_property` (line 122).
- **Writer:** First access triggers `driver.active.get_benchmarker()`.
- **Race scenario:** Two threads access `self.do_bench` concurrently for the
  first time. Both execute the getter, both call
  `driver.active.get_benchmarker()`. Both write to `self.__dict__`. Under
  free-threaded CPython, `cached_property` has no lock (removed in 3.12).
  The getter is pure and the result is identical, so duplicate execution is
  benign — same pattern as `KernelParam` cached properties in the jit audit.
  Tier 2. Minor because the result is functionally identical.

### 7. `check_disk_cache` — compound races on shared instance state (Significant)

- **Shared state:** `self.cache[tuning_key]` (line 198), `self.configs_timings`
  (line 199), the disk cache itself.
- **Race scenario:** Two threads with the same tuning key both call
  `check_disk_cache`. Both miss the in-memory cache. Both check the disk
  cache. If neither finds a disk entry, both run `bench_fn()`, both write to
  disk. If one finds a disk entry while the other is benchmarking, the
  in-memory state (`self.cache`, `self.configs_timings`) gets written from
  two paths with different data. The disk-write path at line 203-209 reads
  `self.configs_timings` which may have been set by the other thread's
  `benchmark()` (issue #3). Tier 2.
- **Relationship to other issues:** This is a compound issue. It requires the
  cache-miss TOCTOU from issue #1 to allow two threads into `check_disk_cache`
  concurrently, and the `self.configs_timings` clobber from issue #3 to
  produce cross-key contamination. The distinct element here is that the
  contaminated data is serialized to disk (line 203-209), making the corruption
  persistent across process restarts. Fixing either #1 (per-key sync) or #3
  (local `configs_timings`) would break the chain, but both are needed for a
  complete fix.

## Near-miss notes

- **`self.configs` list** (line 35): Set during `__init__`, read-only
  thereafter. Not a race.

- **`self.keys`, `self.arg_names`**: Set during `__init__`, immutable. Safe.

- **`Config` objects**: Immutable after construction (`kwargs`, `num_warps`,
  etc. are set in `__init__` and never mutated). Safe to share.

- **`Heuristics` class** (line 449): Stateless wrapper — `run()` only reads
  `self.values` and `self.arg_names` (both set in `__init__`), delegates to
  `self.fn.run()`. No mutable instance state. Safe.

- **`knobs.autotuning.print` / `knobs.autotuning.cache` reads**: Single loads,
  no re-read TOCTOU. Stale reads produce cosmetic effects (extra/missing
  print) or wrong-but-stable `cache_results` at init time. Benign.
