# `self.best_config` and `self.bench_time` stale across concurrent calls

- **Status:** Open
- **Severity:** Minor
- **Component:** `python/triton/runtime/autotuner.py`

- **Shared state:** `self.best_config` (line 245) and `self.bench_time`
  (line 231) — written per-call on the shared `Autotuner` instance.
- **Writer(s):** `run()` writes `self.best_config` at line 245 for every call.
  `benchmark()` writes `self.bench_time` at line 231 on cache misses.
- **Reader(s):** The diagnostic print at line 247-248. No other internal or
  external readers within Triton.
- **Race scenario:** Thread A benchmarks key K1 and sets both attributes.
  Thread B benchmarks key K2 and overwrites them. Thread A's print statement
  reports Thread B's config and timing. The actual kernel launch uses the
  local `config` variable (line 242), so correctness is unaffected.
- **Consequences:** Wrong diagnostic output only. No effect on kernel
  selection or execution.
- **Suggested fix:** Use local variables for the print statement instead of
  reading back from `self`.
