# `Autotuner.run()` cache-miss TOCTOU on shared benchmark state

- **Status:** Open
- **Severity:** SEVERE
- **Component:** `python/triton/runtime/autotuner.py`

- **Shared state:** `self.cache`, plus per-call benchmark scratch state stored
  on the shared `Autotuner` instance: `self.nargs`, `self.restore_copies`,
  `self.configs_timings`, `self.bench_time`, and `self.best_config`.
- **Writer(s):**
  - `run()` writes `self.nargs = dict(zip(...))`, checks `if key not in
    self.cache`, benchmarks on a miss, then writes `self.cache[key] = ...`.
  - The default restore hooks write and clear `self.restore_copies`.
  - Benchmark / disk-cache paths write `self.configs_timings` and
    `self.bench_time`.
- **Reader(s):**
  - `run()` reads `self.cache` for the miss check and later reads the chosen
    config back.
  - `_bench()` builds `full_nargs = {**self.nargs, **current}` and passes it to
    `config.pre_hook`, `self.pre_hook`, and `self.post_hook`.
  - `prune_configs()` reads `self.nargs` for `early_config_prune` and
    `perf_model`.
  - Disk-cache serialization reads `self.configs_timings`.
- **Race scenario:** Thread A and Thread B call the same autotuned kernel with
  the same tuning key. Both observe `key not in self.cache` and both enter the
  benchmark path. While Thread A is in `_bench()`, Thread B overwrites
  `self.nargs`, so Thread A's hooks / pruning logic see Thread B's
  tensors/descriptors instead of its own. With `restore_value` /
  `reset_to_zero`, one thread can zero, snapshot, or restore the wrong tensor,
  or fail to restore a benchmarked tensor at all. Concurrent writes to
  `self.restore_copies` can also make one thread restore from another thread's
  clone, or hit `KeyError` / PyTorch `RuntimeError` when the dict is cleared or
  the cross-thread `copy_()` is incompatible. `config.pre_hook` races are worse
  because they can mutate the wrong descriptor-like launch state for a
  benchmark run. Even when hooks are benign, both threads still do duplicate
  benchmark/compile/launch work, and `self.configs_timings` can be serialized
  under the wrong disk-cache key.
- **Suggested fix:** Do not keep per-call benchmark state on `self`. Pass
  `nargs` into `_bench()` / pruning logic, keep restore snapshots and timings
  local to the call, and add per-key synchronization so only one thread
  benchmarks and populates a given cache key while others wait for the result.
  This matches the fix direction in upstream PR `triton-lang/triton#8437`.