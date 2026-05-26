# `JITFunction.pre_run_hooks` unsynchronized iteration during concurrent mutation

- **Status:** Open
- **Severity:** MED
- **Component:** `runtime/jit.py`
- **Tier:** 3
- **Patch:** [`pre-run-hooks-unsynchronized-iteration.patch`](pre-run-hooks-unsynchronized-iteration.patch)

- **Shared state:** `self.pre_run_hooks` -- a plain `list` of user-supplied
  callables, initialized in `JITFunction.__init__`, never rebound
- **Writer(s):** `add_pre_run_hook` calls `list.append` with no lock
- **Reader(s):** `run()` iterates the list with
  `for hook in self.pre_run_hooks:` on every kernel launch
- **Race scenario:** Thread A is iterating `self.pre_run_hooks` in `run()`
  while Thread B calls `add_pre_run_hook`. Per the Python thread safety
  documentation, iteration while modifying a list is not thread-safe -- the
  observed element sequence is undefined. Previously registered hooks may be
  skipped or visited twice on Thread A's in-flight iteration.
- **Suggested fix:** Copy-on-write: `add_pre_run_hook` builds a new list and
  rebinds `self.pre_run_hooks` (under a writer lock to prevent lost updates);
  `run()` iterates the snapshot it loaded. Attribute rebinding is atomic, so
  readers always see a complete list. No lock needed on the hot path.
