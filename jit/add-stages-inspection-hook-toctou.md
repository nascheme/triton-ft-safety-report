# `knobs.runtime.add_stages_inspection_hook` TOCTOU in `run()`

- **Status:** Open
- **Patch:** `add-stages-inspection-hook-toctou.patch` (single load of the hook slot into a local before the None-check and call)
- **Severity:** Significant
- **Component:** `runtime/jit.py`

- **Shared state:** `knobs.runtime.add_stages_inspection_hook` -- a
  process-global slot holding `None` or a callable
- **Writer(s):** Any thread mutating the slot (plain attribute, no
  synchronization)
- **Reader(s):** `JITFunction.run` (jit.py:727-729) reads the slot **twice**:
  once for the `is not None` check and once to call it
- **Race scenario:** Thread A evaluates
  `knobs.runtime.add_stages_inspection_hook is not None` as `True`. Thread B
  clears the slot to `None`. Thread A performs the second read and calls
  `None()` -> `TypeError`. If the slot is swapped to a different callable
  instead of cleared, the specialization key incorporates a hash from a hook
  that is no longer installed, producing an inconsistent cache key.
- **Suggested fix:** Single-load into a local variable so the `is not None`
  check and the call use the same captured reference.
