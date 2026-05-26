# `knobs.runtime.add_stages_inspection_hook` TOCTOU in `compile()`

- **Issue-Id:** FT008
- **Status:** Open
- **Severity:** MED
- **Component:** `python/triton/compiler/compiler.py`
- **Tier:** 3
- **Patch:** [`add-stages-inspection-hook-toctou.patch`](add-stages-inspection-hook-toctou.patch)

- **Shared state:** `knobs.runtime.add_stages_inspection_hook` — a
  plain `Optional[PipelineStagesHook]` slot on the process-global
  `knobs.runtime` object.
- **Writer(s):** Any thread assigning to the slot (e.g. profiling
  framework clearing or swapping the hook).
- **Reader(s):** `compile()` on every cache-missing compile.
- **Race scenario:** The slot is loaded twice — once for `is not None`
  and once to invoke it. Thread A checks `is not None` → True. Thread B
  clears the slot to `None`. Thread A re-reads and calls `None()` →
  `TypeError`. Alternatively, Thread B swaps to a different hook;
  Thread A invokes the new hook, folding its `inspect_stages_key` into
  the filesystem cache key under a stale context.
- **Suggested fix:** Single load into a local:
  ```python
  hook = knobs.runtime.add_stages_inspection_hook
  if hook is not None:
      inspect_stages_key, inspect_stages_hash = hook()
      key += inspect_stages_key
  ```

Same pattern as [`jit/add-stages-inspection-hook-toctou.md`](../jit/add-stages-inspection-hook-toctou.md)
on the launch path; both sites need independent fixes.
