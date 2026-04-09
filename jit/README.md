# runtime/jit.py Free-Threading Issues

Issues in `python/triton/runtime/jit.py` affecting free-threaded Python 3.14t.

## Issues

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| 1 | Significant | function_registry | [`_triton_jit_function_registry` publishes partially-initialized `JITFunction`](function-registry-race.md) |
| 2 | SEVERE | device_caches | [`JITFunction.device_caches` defaultdict auto-vivification race](device-caches-race.md) |
| 3 | Not a bug | create_binder | `create_binder` sets instance attributes on `self` redundantly (dict ops are thread-safe in free-threaded CPython; values are always identical) |
| 4 | Significant | kernel_cache | [`kernel_cache` / `kernel_key_cache` TOCTOU causes duplicate compilation](kernel-cache-toctou.md) |
| 5 | Not a bug | KernelParam | `KernelParam` `@cached_property` fields — getters are pure/deterministic and `__dict__` writes are thread-safe in free-threaded CPython; worst case is benign duplicate computation |
| 6 | Significant | used_global_vals | [`JITCallable.used_global_vals` unsynchronized read skips global-changed safety check](used-global-vals-unsynchronized-read.md) |
| 7 | Significant | pre_run_hooks | [`JITFunction.pre_run_hooks` unsynchronized iteration during concurrent mutation](pre-run-hooks-unsynchronized-iteration.md) |
| 8 | (covered by #4) | compute_cache_key | [`compute_cache_key` read-then-write race on `kernel_key_cache`](kernel-cache-toctou.md) |
| 9 | Significant | async_compile | [`_do_compile` async path — `AsyncCompileMode`/`FutureKernel` races](async-compile-races.md) |
| 10 | Not a bug | hash_placeholder | [`JITCallable.hash` placeholder visibility — no direct readers exist](hash-placeholder-not-a-bug.md) |
| 11 | Significant | _unsafe_update_src | [`JITCallable._unsafe_update_src` unsynchronized hash invalidation](unsafe-update-src-race.md) |

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
  Concurrent `dict.get()` and `dict.__setitem__()` on the same dict is the
  classic compound race under free-threading. Also, the `finalize_compile`
  callback (line 877) writes `kernel_cache[key]` from the async executor
  thread. Tier 2. Written up in
  [kernel-cache-toctou.md](kernel-cache-toctou.md).

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
