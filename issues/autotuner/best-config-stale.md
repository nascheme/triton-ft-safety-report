# `self.best_config` and `self.bench_time` stale across concurrent calls

- **Issue-Id:** FT001
- **Status:** Open
- **Rank:** Low
- **Component:** `python/triton/runtime/autotuner.py`
- **Patch:** [`best-config-stale.patch`](best-config-stale.patch)

- **Shared state:** `self.best_config` and `self.bench_time` — written
  per-call on the shared `Autotuner` instance.
- **Writer(s):** `run()` writes `self.best_config` for every call.
  `benchmark()` writes `self.bench_time` on cache misses.
- **Reader(s):** The diagnostic print at the end of `run()`. No other
  internal or external readers within Triton.
- **Race scenario:** Thread A benchmarks key K1 and sets both attributes.
  Thread B benchmarks key K2 and overwrites them. Thread A's print statement
  reports Thread B's config and timing. The actual kernel launch uses the
  local `config` variable, so correctness is unaffected.
- **Consequences:** Wrong diagnostic output only. No effect on kernel
  selection or execution.
- **Suggested fix:** Use local variables for the print statement instead of
  reading back from `self`.
- **Note:** `self.bench_time` and `self.best_config` are retained on the
  assumption that they are part of the public `Autotuner` API and cannot be
  removed. Nothing in the Triton codebase reads `self.bench_time`;
  `self.best_config` is read by example scripts
  (`python/examples/gluon/03-matmul-multicta.py`,
  `python/examples/gluon/04-2cta-block-scale-matmul.py`), confirming it is
  observable by user code.
