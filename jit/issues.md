# runtime/jit.py Free-Threading Issues

Issues in `python/triton/runtime/jit.py` affecting free-threaded Python 3.14t.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | Significant | function_registry | 1 | [`_triton_jit_function_registry` publishes partially-initialized `JITFunction`](function-registry-race.md) |
| 2 | SEVERE | device_caches | 2 | [`JITFunction.device_caches` defaultdict auto-vivification race](device-caches-race.md) |
| 4 | Significant | kernel_cache | 2 | [`kernel_cache` / `kernel_key_cache` TOCTOU causes duplicate compilation](kernel-cache-toctou.md) |
| 6 | Significant | used_global_vals | 2 | [`JITCallable.used_global_vals` unsynchronized read skips global-changed safety check](used-global-vals-unsynchronized-read.md) |
| 7 | Significant | pre_run_hooks | 3 | [`JITFunction.pre_run_hooks` unsynchronized iteration during concurrent mutation](pre-run-hooks-unsynchronized-iteration.md) |
| 8 | (covered by #4) | compute_cache_key | 2 | [`compute_cache_key` read-then-write race on `kernel_key_cache`](kernel-cache-toctou.md) |
| 9 | Significant | async_compile | 2 | [`_do_compile` async path — `AsyncCompileMode`/`FutureKernel` races](async-compile-races.md) |
| 11 | Significant | _unsafe_update_src | 3 | [`JITCallable._unsafe_update_src` unsynchronized hash invalidation](unsafe-update-src-race.md) |
| 12 | Significant | add_stages_inspection_hook | 3 | [`knobs.runtime.add_stages_inspection_hook` TOCTOU in `run()`](add-stages-inspection-hook-toctou.md) |

## Triage notes

Candidates surfaced during an initial pass over `jit.py`. Issues that were
promoted to their own files are cross-linked; the remaining notes are kept
here as a record of the original triage reasoning.

### 2. `JITFunction.device_caches` — defaultdict auto-vivification race

- **Shared state:** `self.device_caches = defaultdict(self.create_binder)` at
  `jit.py:790`.
- **Writer:** Auto-vivification on first access for each device key.
- **Reader/Writer:** `run()` (line 720), `_do_compile()` (line 860),
  `preload()` (line 816).
- **Concern:** `defaultdict.__getitem__` with a missing key calls the
  factory, creates the value, and inserts it — the sequence is not atomic.
  Two threads launching the same kernel on the same device for the first
  time will race. Tier 2. Written up in
  [device-caches-race.md](device-caches-race.md).

### 3. `create_binder` sets instance attributes on `self` non-atomically

- **Shared state:** `self.CompiledKernel`, `self.compile`, `self.ASTSource`
  at lines 678-679.
- **Concern:** Called from the `device_caches` defaultdict factory. Each
  new device triggers this, overwriting these attributes. Two threads
  adding different devices concurrently race on these `self.*` writes.
  Thread A could read `self.ASTSource` set by thread B (wrong backend).
  Tier 2.
- **Resolution:** Not a bug on its own. Dict ops on `self.__dict__` are
  thread-safe in free-threaded CPython, and the values written are always
  identical for a given `JITFunction`, so any interleaving yields the same
  observable state.

### 4. `kernel_cache` and `kernel_key_cache` — unprotected dict read/write

- **Writer:** `_do_compile()` lines 877, 882, 885; `compute_cache_key()`
  line 594.
- **Reader:** `run()` lines 731-732.
- **Concern:** These are plain dicts inside the `device_caches` tuple.
  The issue is the check-then-act TOCTOU that spans multiple dict operations,
  not the individual dict ops (which are internally thread-safe in
  free-threaded CPython). Also, the `finalize_compile` callback (line 877)
  writes `kernel_cache[key]`. Tier 2. Written up in
  [kernel-cache-toctou.md](kernel-cache-toctou.md).

  On the async path there is a further ordering hazard: after
  `async_mode.submit` returns, the executor may complete the compile and
  another thread may call `FutureKernel.result()` -> `finalize_compile`,
  writing `kernel_cache[key] = real_kernel` **before** the submitting
  thread's placeholder write at jit.py:882 executes. The placeholder then
  overwrites the real kernel. Subsequent readers get a `FutureKernel` whose
  `finalize_compile` already fired; functionally it still works (the
  `__getattr__` -> `result()` path returns the cached kernel), but the
  cache holds a wrapper instead of the real kernel object.

### 5. `KernelParam` `@cached_property` fields — not a bug

- **Fields:** `name`, `annotation`, `annotation_type`, `is_constexpr`,
  `is_const` at lines 309-337. `KernelParam` instances are shared via
  `JITFunction.params` (line 783), so concurrent first-access from multiple
  threads is realistic under Tier 2.
- **Resolution:** Not a bug under free-threading.
  1. `functools.cached_property` has had no lock since CPython 3.12
     (deliberate removal, python/steering-council#172). This is unchanged
     under free-threading.
  2. All five getters are pure functions of immutable state
     (`self._param.*`) and module-level dicts. `_normalize_ty` (line 273)
     is also pure. Duplicate execution across threads returns identical
     values.
  3. Concurrent writes into an instance `__dict__` are internally
     thread-safe in free-threaded CPython (dict ops are safe).
- **Worst case:** Two threads may both execute the getter and both write
  the same value to `__dict__`. This is benign duplicate computation,
  parallel to issue #3.

### 6. `JITCallable.used_global_vals` — written into

Written up in [used-global-vals-unsynchronized-read.md](used-global-vals-unsynchronized-read.md).

### 7. `JITFunction.pre_run_hooks`

Written up in [pre-run-hooks-unsynchronized-iteration.md](pre-run-hooks-unsynchronized-iteration.md).

### 8. `compute_cache_key` — read-then-write race on `kernel_key_cache`

- **Lines:** 574-595:
  `cache_key = kernel_key_cache.get(key, None) ... kernel_key_cache[key] = cache_key`.
- **Concern:** Classic TOCTOU — two threads check for the same key, both
  find it missing, both compute and write. The duplicate computation is
  benign, but the concurrent `dict.__setitem__` is the real issue (same
  dict as #4). Tier 2, covered by #4
  ([kernel-cache-toctou.md](kernel-cache-toctou.md)).

### 9. `_do_compile` async path — `AsyncCompileMode`/`FutureKernel` races

Written up in [async-compile-races.md](async-compile-races.md). Investigation
downgraded from SEVERE to Significant: the concrete bugs are (1) a TOCTOU in
`AsyncCompileMode.submit` causing duplicate executor submissions for the same
`cache_key`, (2) non-atomic `FutureKernel.result` causing duplicate
`finalize_compile` and therefore duplicate `jit_post_compile_hook`
invocations, and (3) an ordering hazard where the placeholder write at
jit.py:882 can overwrite an already-finalized entry. The "double-write from
an executor thread" framing in the original triage note is not quite right —
`finalize_compile` actually runs on whichever thread calls
`FutureKernel.result()` (either `AsyncCompileMode.__exit__` or a
`__getattr__` triggered by `kernel_cache`'s placeholder consumer), not on
the executor.

### 10. `JITCallable.hash` — visible before `used_global_vals` is populated

Investigated and resolved as **not a bug**. See
[hash-placeholder-not-a-bug.md](hash-placeholder-not-a-bug.md) — no direct
reader of `self.hash` exists outside the `cache_key` property, so the
placeholder never escapes the `_hash_lock` critical section.

### 11. `JITCallable._unsafe_update_src` — unsynchronized hash invalidation

Written up in [unsafe-update-src-race.md](unsafe-update-src-race.md).
Confirmed as a Tier 3 ordering bug: Thread A's in-flight `cache_key`
finalization can clobber Thread B's `self.hash = None` write, leaving
`self.hash` pointing at the old-src hash after `self._src` has been
replaced. Fix is to take `self._hash_lock` inside `_unsafe_update_src`.

There is a secondary hazard: `cache_key` reads `self._src` twice -- once
via `self.src` (line 510, for the dependencies finder) and once via
`self.parse()` (line 511, which re-reads `self._src`). If
`_unsafe_update_src` replaces `_src` between those two reads, the
dependencies finder walks the new-src AST while holding the old src string.
The resulting hash is derived from an inconsistent pair. This is a
second-order consequence of the same missing-lock root cause.

### 12. `knobs.runtime.add_stages_inspection_hook` — TOCTOU in `run()`

Written up in
[add-stages-inspection-hook-toctou.md](add-stages-inspection-hook-toctou.md).
The `run()` launch path reads the slot twice at jit.py:727-729 — once
for `is not None` and once to call it. A concurrent clear of the slot
between the two reads turns the second read into `None()` and raises
`TypeError`. A concurrent swap yields a cache key derived from a hook
that is no longer current. Single-load into a local fixes the crash
case. Tier 3.

## Near-miss notes

Additional candidates examined during a second pass and decided against
reporting. Kept here so a future auditor does not have to redo the work.

- **`DependenciesFinder._update_hash` multi-read on `func.used_global_vals`**
  (jit.py:103-111). Reads `func.used_global_vals` several times before
  calling `func.cache_key` at line 113. The attribute is only assigned
  inside `cache_key` under `func._hash_lock`, goes atomically from `{}`
  to a finalized dict, and `_unsafe_update_src` (issue #11) does not
  reset it — so the pre-`cache_key` reads see either the empty dict
  (empty intersection, no-op) or a stable populated dict. Residual
  hazard is covered by #11's `_hash_lock` discussion.

- **`create_binder` writes to `self.CompiledKernel` / `self.compile` /
  `self.ASTSource`** (jit.py:678-680) racing with reads in `_do_compile`
  (jit.py:865, 874). Same "benign redundant write" pattern as the
  not-a-bug entry #3 — the assigned values are module-level imports and
  are bitwise identical regardless of which device triggered the
  binder, so any interleaving is observationally equivalent.

- **`_triton_jit_function_registry` lookup in `preload()`**
  (jit.py:831-832) is an `in`-then-`[]` check-then-get. Registry is
  append-only (no deletions anywhere in the file), so the `[]` cannot
  raise `KeyError` even if the `in` check races with an insert.

- **`finalize_compile` closure writing to `kernel_cache`** (jit.py:877).
  Same dict object stored in the `device_caches` tuple — every write
  site is covered by the kernel-cache TOCTOU writeup (#4) and the
  async-compile writeup (#9); this is not an additional site.

- **`knobs.runtime.debug` / `knobs.compilation.instrumentation_mode`
  reads** at jit.py:709-710. Each is a single atomic attribute load
  with no re-read, so there is no TOCTOU. Stale reads here are the
  benign-debug-flag pattern CLAUDE.md carves out.

- **`knobs.runtime.launch_enter_hook` / `launch_exit_hook` reads** at
  jit.py:762. Passed directly into the native launch call as values;
  no re-read, so no TOCTOU. Mutation of these slots is a knobs-level
  concern, not a jit.py race.
