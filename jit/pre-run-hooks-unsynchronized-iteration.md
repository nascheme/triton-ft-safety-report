# `JITFunction.pre_run_hooks` unsynchronized iteration during concurrent mutation

- **Status:** Open
- **Severity:** Significant
- **Component:** `runtime/jit.py`

- **Shared state:** `self.pre_run_hooks` -- a plain `list` of user-supplied
  callables, initialized at jit.py:804, never rebound
- **Writer(s):** `add_pre_run_hook` (jit.py:663-669) calls `list.append`
  with no lock
- **Reader(s):** `run()` (jit.py:716-718) iterates the list with `for hook in
  self.pre_run_hooks:` on every kernel launch
- **Race scenario:** Thread A is iterating `self.pre_run_hooks` in `run()`
  while Thread B calls `add_pre_run_hook`. Per the Python thread safety
  documentation, iteration while modifying a list is not thread-safe -- the
  observed element sequence is undefined. Previously registered hooks may be
  skipped or visited twice on Thread A's in-flight iteration.
- **Suggested fix:** Copy-on-write: `add_pre_run_hook` builds a new list and
  rebinds `self.pre_run_hooks` (under a writer lock to prevent lost updates);
  `run()` iterates the snapshot it loaded. Attribute rebinding is atomic, so
  readers always see a complete list. No lock needed on the hot path.
