# `knobs.runtime.add_stages_inspection_hook` TOCTOU in `compile()`

- **File:** `python/triton/compiler/compiler.py`
- **Severity:** Significant
- **Tier:** 3 (configuration mutation concurrent with compilation)

This is the same TOCTOU pattern as
[`jit/add-stages-inspection-hook-toctou.md`](../jit/add-stages-inspection-hook-toctou.md)
but on the `compile()` side. The jit.py site reads the hook on the
**launch** path (specialization string); this site reads it on the
**compile** path (filesystem cache key). Both sites are independently
racey and both need the single-load fix.

## Shared state

`knobs.runtime.add_stages_inspection_hook` — a plain module-level
`Optional[PipelineStagesHook]` slot on the `knobs.runtime` object
(`knobs.py:480`). No synchronization. Users install/clear it to hook
into pipeline-stage inspection.

## Writer

Any thread assigning to `knobs.runtime.add_stages_inspection_hook` —
typically a profiling/inspection framework clearing or swapping the
hook when a window ends.

## Reader

`compile()` at `compiler.py:247-249`, on every cache-missing compile:

```python
if knobs.runtime.add_stages_inspection_hook is not None:
    inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
    key += inspect_stages_key
```

The slot is loaded **twice** — once for the `is not None` check and
once to invoke it. `compile()` is reached from
`JITFunction._do_compile` whenever the in-memory `kernel_cache` misses,
which under Tier 2/Tier 3 can easily run concurrently with a thread
mutating the hook slot.

## Race scenario

### Scenario A — spurious `TypeError` on the compile path

1. Thread A enters `compile()` for a cache-missing kernel and evaluates
   `knobs.runtime.add_stages_inspection_hook is not None` as `True`
   (line 247).
2. Thread B clears the hook:
   `knobs.runtime.add_stages_inspection_hook = None`.
3. Thread A re-reads the slot at line 248 and calls `None()`, raising
   `TypeError: 'NoneType' object is not callable`.
4. Compilation fails. The exception propagates out of
   `JITFunction._do_compile` → `JITFunction.run`, aborting the user's
   kernel launch with an error unrelated to their kernel. The failure
   is timing-dependent and effectively unreproducible.

### Scenario B — filesystem cache-key divergence on swap

`key` at line 246 is the seed for the filesystem cache hash computed at
line 250 (`hash = hashlib.sha256(key.encode(...))`), which becomes the
directory key passed to `get_cache_manager(hash)` at line 251. So the
hook's `inspect_stages_key` is baked into the on-disk cache key.

1. Hook H1 is installed. Thread A enters `compile()` and the `is not
   None` check at 247 returns `True` while H1 is still the current hook.
2. Thread B swaps the slot: `knobs.runtime.add_stages_inspection_hook = H2`.
3. Thread A's second read at 248 returns H2, so `key` is extended with
   H2's `inspect_stages_key`. The resulting filesystem cache directory
   is keyed with H2's key even though at the moment of entry the
   "logically installed" hook was H1.
4. A subsequent `compile()` call (in the same process or another), now
   consistently observing H2, will correctly key on H2 and hit/populate
   the same directory — but cache entries generated while H1 was
   installed are orphaned under H2's key. Cache lookups and on-disk
   state are no longer consistent with the hook history that actually
   ran.

Unlike the launch-path variant, the consequence here is not just an
in-memory specialization list mismatch: the **persistent on-disk cache
layout** depends on a value that was observed torn across a concurrent
writer. Subsequent runs of the process inherit that inconsistency.

## Consequence

1. **Crash on the compile hot path** when a concurrent thread clears
   the hook. Surfaces as an unrelated `TypeError` from user kernel
   launches.
2. **Filesystem cache-key divergence** when the hook is swapped: the
   on-disk compile artifact directory is keyed under a hook that was
   not the one the surrounding logic believed was current, producing
   orphaned cache entries that persist across process restarts.
3. **Independent of the jit.py fix.** Fixing only the jit.py site
   (`runtime/jit.py:727-729`) leaves this site racey. `compile()` is
   on a distinct code path (cache-missing compile, also reachable from
   `JITFunction.warmup` and external callers of
   `triton.compiler.compile`), so it needs its own single-load fix.

This is in scope under the CLAUDE.md knob-read carve-out — it affects
compilation correctness and cache-key behavior, not just debug-flag
staleness.

## Suggested fix

Single load into a local, matching the jit.py fix:

```python
inspection_hook = knobs.runtime.add_stages_inspection_hook
if inspection_hook is not None:
    inspect_stages_key, inspect_stages_hash = inspection_hook()
    key += inspect_stages_key
```

This prevents the `TypeError` crash and guarantees that whichever hook
is observed at the `is not None` check is the same one whose key is
folded into `hash`. The swap-vs-clear cross-call inconsistency is not
fully resolved by this change — a complete fix would snapshot relevant
configuration at the top of `compile()` — but the single-load is
sufficient to make one `compile()` invocation self-consistent.

## Related

- [`jit/add-stages-inspection-hook-toctou.md`](../jit/add-stages-inspection-hook-toctou.md)
  — same pattern on the launch path in `JITFunction.run`.
- Issue #3 in [`compiler/README.md`](README.md) — the
  `kernel_load_start_hook` / `kernel_load_end_hook` TOCTOUs in
  `_init_handles` share the same `is not None` + re-read + call shape
  and should be fixed together.
- `inspect_stages_hash` is assigned at line 248 but never read within
  `compile()` (unlike jit.py:729 where it is folded into the
  specialization string). That is a separate dead-store observation,
  not part of the race.
