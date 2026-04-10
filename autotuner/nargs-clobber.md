# `self.nargs` clobbered by concurrent `run()` calls

- **Status:** Open
- **Severity:** Significant
- **Component:** `python/triton/runtime/autotuner.py`

- **Shared state:** `self.nargs` — a `dict` set on the shared `Autotuner`
  instance at the start of every `run()` (line 213) and `warmup()` (line 288),
  cleared at exit (lines 257, 296).
- **Writer(s):** Every `run()` and `warmup()` caller overwrites `self.nargs`
  unconditionally with its own arguments.
- **Reader(s):** `_bench()` (line 143: `full_nargs = {**self.nargs, ...}`),
  `prune_configs()` (line 263: `early_config_prune(self.configs, self.nargs, ...)`
  and line 278: `self.perf_model(**self.nargs, ...)`), `benchmark()` inner
  function (line 233), and `run()` post-benchmark (line 250).
- **Race scenario:** Thread A calls `run(x_ptr, 1024)`, setting `self.nargs`
  to `{name: x_ptr, N: 1024}`. Thread B calls `run(y_ptr, 2048)`, overwriting
  `self.nargs` with `{name: y_ptr, N: 2048}`. Thread A's in-progress
  `_bench()`, `prune_configs()`, or hook callbacks now use Thread B's arguments.
  In another interleaving, Thread A finishes and sets `self.nargs = None`;
  Thread B's still-running `_bench()` reads `None` and crashes on
  `{**self.nargs, ...}`.
- **Consequences:** Wrong arguments passed to `config.pre_hook`,
  `self.pre_hook`, `self.post_hook`, `early_config_prune`, and `perf_model`.
  Hooks may mutate wrong tensors or descriptors. `None` reads cause
  `TypeError`.
- **Suggested fix:** Pass `nargs` as a local variable through `_bench()`,
  `prune_configs()`, and `benchmark()` instead of storing it on `self`. This
  is the approach taken in upstream PR `triton-lang/triton#8437`.
