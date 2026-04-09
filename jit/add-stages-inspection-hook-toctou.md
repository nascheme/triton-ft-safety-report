# `knobs.runtime.add_stages_inspection_hook` TOCTOU in `run()`

- **File:** `python/triton/runtime/jit.py`
- **Severity:** Significant
- **Tier:** 3 (configuration mutation concurrent with kernel launch)

## Shared state

`knobs.runtime.add_stages_inspection_hook` — a process-global slot on the
`knobs.runtime` object holding either `None` or a callable. Users set it
to install an inspection hook that contributes a per-kernel pipeline
specialization key/hash.

## Writer

Any thread mutating `knobs.runtime.add_stages_inspection_hook`. The slot
has no synchronization; it is a plain attribute on a module-level object.

## Reader

`JITFunction.run` (jit.py:727-729), on every kernel launch:

```python
if knobs.runtime.add_stages_inspection_hook is not None:
    inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
    specialization.append(f'("custom_pipeline", {inspect_stages_hash})')
```

This reads the slot **twice**: once for the `is not None` check, and
once to call it. Between the two reads, any other thread may rebind the
slot.

## Race scenario

Thread A is in `JITFunction.run` and has just evaluated
`knobs.runtime.add_stages_inspection_hook is not None` as `True`.

Thread B, executing unrelated configuration logic, does
`knobs.runtime.add_stages_inspection_hook = None` (for example, clearing
an installed hook after a profiling window ends).

Thread A then performs the second read,
`knobs.runtime.add_stages_inspection_hook()`, which now returns `None`,
and raises `TypeError: 'NoneType' object is not callable`. The kernel
launch fails with an error that is unrelated to the kernel itself and is
effectively impossible to reproduce.

Even when the second read still sees a callable, the two reads may
observe **different** callables if the hook is swapped (not just
cleared). The specialization then incorporates a hash from a hook that
is no longer "the installed one," and the resulting cache key is
inconsistent with the runtime configuration visible to the rest of the
system — the next launch with the new hook then compiles a separate
cache entry under the same logical configuration. This is a cache-
behavior consequence, not just staleness.

## Consequence

1. **Spurious `TypeError` crash** on the kernel launch hot path when a
   concurrent thread clears the hook. Tier 3 configuration mutation is
   an explicitly supported use case, so this is a real failure mode.
2. **Cache-key divergence** when the hook is swapped rather than
   cleared: the compiled kernel is keyed by the outgoing hook's hash
   while the runtime believes the incoming hook is active.

This hits the CLAUDE.md in-scope carve-out for knob reads — it is not
benign debug-flag staleness. It affects launch correctness (crash) and
cache-key computation.

## Suggested fix

Single load into a local variable:

```python
inspection_hook = knobs.runtime.add_stages_inspection_hook
if inspection_hook is not None:
    inspect_stages_key, inspect_stages_hash = inspection_hook()
    specialization.append(f'("custom_pipeline", {inspect_stages_hash})')
```

This eliminates the TOCTOU: the `is not None` check and the call both
operate on the same captured reference. A concurrent writer may still
change the slot, but Thread A's launch consistently uses the hook that
was installed at the moment of the load.

The swap-vs-clear inconsistency on the cache key is not fully resolved
by this change — a snapshot of configuration state at launch entry
would be a more complete fix — but a single-load is enough to prevent
the crash and to make the launch self-consistent.
