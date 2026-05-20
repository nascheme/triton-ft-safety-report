# `knobs.runtime.add_stages_inspection_hook` TOCTOU in `compile()`

- **Status:** Open
- **Patch:** `add-stages-inspection-hook-toctou.patch` (single load of the hook slot into a local before the None-check and call)
- **Severity:** Significant
- **Component:** `python/triton/compiler/compiler.py`
- **Tier:** 3

- **Shared state:** `knobs.runtime.add_stages_inspection_hook` — a
  plain `Optional[PipelineStagesHook]` slot on the process-global
  `knobs.runtime` object.
- **Writer(s):** Any thread assigning to the slot (e.g. profiling
  framework clearing or swapping the hook).
- **Reader(s):** `compile()` at lines 247–249, on every cache-missing
  compile.
- **Race scenario:** The slot is loaded twice — once for `is not None`
  (line 247) and once to invoke it (line 248). Thread A checks
  `is not None` → True. Thread B clears the slot to `None`. Thread A
  re-reads and calls `None()` → `TypeError`. Alternatively, Thread B
  swaps to a different hook; Thread A invokes the new hook, folding
  its `inspect_stages_key` into the filesystem cache key under a
  stale context.
- **Suggested fix:** Single load into a local:
  ```python
  hook = knobs.runtime.add_stages_inspection_hook
  if hook is not None:
      inspect_stages_key, inspect_stages_hash = hook()
      key += inspect_stages_key
  ```

Same pattern as [`jit/add-stages-inspection-hook-toctou.md`](../jit/add-stages-inspection-hook-toctou.md)
on the launch path; both sites need independent fixes.
