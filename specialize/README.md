# python/src/specialize.cc Free-Threading Issues

## Issues

| # | Severity | Tier | Component | Issue |
|---|----------|------|-----------|-------|
| 1 | SEVERE  | 1 | `init_globals` / `init_called` | [`init_globals()` TOCTOU: plain-bool guard allows concurrent lazy init of all file-scope globals](init-globals-toctou.md) |
| 2 | SEVERE  | 2 | `dtype_ptr2str`                | [`dtype_ptr2str` `std::unordered_map` mutated on the hot path without synchronization](dtype-ptr2str-unordered-map-race.md) |
| 3 | SEVERE  | 2 | `dtype2str`                    | [`dtype2str` `std::unordered_map` mutated on the hot path without synchronization](dtype2str-unordered-map-race.md) |
| 4 | SEVERE  | 1 | `type_handler_cache`           | [`type_handler_cache` read during / before its population races init and specialize-dispatch](type-handler-cache-init-race.md) |
| 5 | Significant | 1 | `init_called` memory ordering | `init_called` is a non-atomic `bool`; readers can observe `true` before dependent writes — subsumed by issue #1, see triage notes |
| 6 | Minor   | 1 | `torch_tensor_cls` lazy probe  | `torch.Tensor` is only discovered if `torch` is already imported at first specialize call — performance-only note, no separate issue file or patch, see triage notes |

Issues in `python/src/specialize.cc` — the native argument-specialization
helper that backs `JITFunction` specialization. This is the highest-priority
C++ file per `CLAUDE.md`: it has lazy process-global initialization guarded
only by a plain boolean, and several process-global native caches populated
and read on the hot specialization path.

Individual issue files exist for the SEVERE items: `init-globals-toctou.md`,
`dtype-ptr2str-unordered-map-race.md`, `dtype2str-unordered-map-race.md`, and
`type-handler-cache-init-race.md`. The last is resolved by
`init-globals-toctou.patch` and has no separate patch. Issue #5 is subsumed
by issue #1 and does not get its own issue file or patch. Issue #6 is Minor
and kept in the table for traceability only — no issue file, no patch.

See the triage notes below for the reasoning behind each entry and for items
that were considered and rejected.


## Context

Entry points exposed to Python:

- `native_specialize_impl` (METH_FASTCALL) — called from
  `runtime/jit.py` specialization for every kernel argument on the hot
  dispatch path. Dispatches to `specialize_arg()`.
- `make_tensordesc_args` (METH_FASTCALL) — translates tensordesc args;
  no shared mutable state at file scope, appears local to its inputs.

Process-global mutable state at file scope (anonymous namespace):

- `static bool init_called` — lazy-init guard, plain `bool`, no atomic,
  no lock. Checked in `specialize_impl` (line 575) and set true at the
  end of `init_globals()` (line 158).
- `static PyObject *` class/function/interned-string globals
  (lines 41–71): `constexpr_cls`, `jit_callable_cls`,
  `tensor_descriptor_cls`, `nvidia_tensor_descriptor_cls`,
  `nvidia_tensor_descriptor_im2col_cls`, `amd_tensor_descriptor_cls`,
  `canonicalize_dtype_fn`, `canonicalize_ptr_dtype_fn`,
  `torch_tensor_cls`, the `*_str` interned strings, the `*_attr`
  interned attribute names, and `align_kwarg`.
- `static DtypePtr2Str dtype_ptr2str` (line 73) — a
  `std::unordered_map<std::pair<Py_hash_t, bool>, PyObject*, …>`
  populated lazily inside `handle_tensor` (line 333).
- `static Dtype2Str dtype2str` (line 74) — a
  `std::unordered_map<Py_hash_t, PyObject*>` populated lazily inside
  `specialize_tensordesc` (line 188).
- `static TypeHandlerCache type_handler_cache` (line 75) — a
  `std::unordered_map<PyTypeObject*, TypeHandler>` populated during
  `init_type_handler_cache()` and read on every `specialize_arg()`
  call (line 516).

All three maps are plain `std::unordered_map`, which is **not** thread-safe
for concurrent insert or concurrent insert+lookup.

The file already flags `lru_cache` / `unordered_map` as the prototypical
SEVERE pattern in `CLAUDE.md` — this file is the textbook example.

## Triage notes

### 1. `init_globals()` TOCTOU on `init_called` (SEVERE)

- **Shared state:** every file-scope global listed in the Context section
  above, plus the `init_called` flag itself.
- **Writer:** `init_globals()` imports Python classes/functions, interns
  strings, calls `init_type_handler_cache()`, and finally sets
  `init_called = true` at line 158.
- **Reader:** `specialize_impl()` tests `if (!init_called) init_globals();`
  at line 575 on every call.
- **Race scenario:** Thread A and Thread B both enter `specialize_impl`
  before any successful init. Both observe `init_called == false` and both
  call `init_globals()` concurrently. `init_globals()`:
  - assigns to each `static PyObject *` global from a new reference
    obtained via `import_from` / `intern_from_string` — the second writer
    overwrites the first writer's pointer, leaking one strong reference per
    global (these are intentionally never DECREF'd, but the double-assign
    still loses a reference);
  - calls `PyImport_GetModuleDict` / `PyDict_GetItemString` and
    `PyObject_GetAttrString` concurrently — these use Python C API and
    are safe as long as each thread holds an attached thread state, but
    torch_tensor_cls assignment can still race;
  - calls `init_type_handler_cache()` concurrently — two threads
    inserting into the same `std::unordered_map<PyTypeObject*, …>` is an
    undefined-behavior container race (see issue #4).
- **Additional concern:** `init_globals()` calls Python C API that can raise
  (`import_from`, attribute lookups). On failure it returns `false` and the
  caller returns `nullptr` with the error set, but `init_called` stays
  `false`, so the next call retries init. Any globals that were successfully
  assigned before the failure remain assigned; a re-run therefore leaks
  those already-assigned references again (minor leak, not a correctness
  bug by itself — called out here because it interacts with the concurrency
  story).
- **Fix direction:** use an `std::atomic<bool>` with acquire/release and a
  mutex-protected slow path. That serializes first-use initialization,
  fully publishes the initialized globals before readers proceed, and keeps
  the existing retry-on-failure behavior if an import/lookup raises.
- **Tier:** Tier 1/2 (first call from any thread can trigger it).

### 2. `dtype_ptr2str` — `std::unordered_map` mutated on hot path (SEVERE)

- **Shared state:** `static DtypePtr2Str dtype_ptr2str`
  (`std::unordered_map<std::pair<Py_hash_t,bool>, PyObject*, …>`), file
  scope, never locked.
- **Writer:** `handle_tensor()` at line 333 — on a cache miss calls
  `canonicalize_ptr_dtype_fn` and stores the resulting `PyObject *` under
  a `(dtype_hash, is_const)` key. The stored value is a new reference that
  is intentionally leaked (lives for process lifetime).
- **Reader:** `handle_tensor()` at line 322 (`dtype_ptr2str.find(dsk)`).
  `handle_tensor()` is reached from `specialize_arg()` on the
  `native_specialize_impl` hot path for tensor-like arguments, either via
  the cached `torch.Tensor` type handler or via the fallback `data_ptr`
  attribute probe.
- **Race scenario:** Thread A takes the miss path and inserts into
  `dtype_ptr2str` while Thread B concurrently calls `find` or performs its
  own insert. `std::unordered_map` concurrent insert/insert or insert/find
  is undefined behavior: possible crash, corrupted buckets, or lost insert.
  This sits directly on the per-argument specialization hot path.
- **Severity:** SEVERE — matches the "non-thread-safe container on hot
  path" pattern from `CLAUDE.md` severity guide.
- **Fix direction:** the safe baseline is to protect the entire access with
  a mutex. That is also the simplest patch: lock around the `find` plus the
  miss-path insert. A more refined alternative is a two-phase miss path:
  lock/check, drop the lock before calling `canonicalize_ptr_dtype_fn`, then
  re-lock, re-check, and publish or discard the new value. That avoids
  holding the mutex across the Python call, but adds ownership/refcount
  complexity. Do **not** use an unlocked fast-path `find` with only a locked
  insert path; `unordered_map::find` itself is unsafe while another thread is
  inserting.
- **Tier:** Tier 2 (two threads calling the same or different kernels with
  tensor args).

### 3. `dtype2str` — `std::unordered_map` mutated on hot path (SEVERE)

- **Shared state:** `static Dtype2Str dtype2str`
  (`std::unordered_map<Py_hash_t, PyObject*>`), file scope, never locked.
- **Writer/Reader:** `specialize_tensordesc()` lines 180–190.
- **Race scenario:** Same as issue #2. Reached for every tensordesc
  argument during specialization.
- **Note:** Narrower hot path than #2 (only tensordesc / Gluon descriptor
  args), but the container-safety story is identical.
- **Fix direction:** protect the map with its own mutex, but avoid holding
  that mutex while calling `canonicalize_dtype_fn`. The clean shape is a
  two-phase miss path: lock/check, compute outside the critical section,
  then re-lock, re-check, and either publish the new `PyObject *` or drop
  the temporary result if another thread already inserted an entry.

### 4. `type_handler_cache` — init-time race through lazy init (SEVERE)

- **Shared state:** `static TypeHandlerCache type_handler_cache`
  (`std::unordered_map<PyTypeObject*, TypeHandler>`).
- **Writer:** `init_type_handler_cache()` (called only from
  `init_globals()`).
- **Reader:** `specialize_arg()` line 516, called from `specialize_impl`
  on every specialization.
- **Root cause:** this is not an independent post-init cache race; it is a
  direct consequence of issue #1. The map is intended to be populated once
  during first-use initialization and then treated as immutable.
- **Race scenario A — init vs init:** Two threads concurrently running
  `init_globals()` (see issue #1) concurrently call
  `init_type_handler_cache()`, each doing several
  `type_handler_cache[key] = fn` inserts. Concurrent `unordered_map`
  mutation is UB.
- **Race scenario B — init vs read:** once two threads are already racing in
  `init_globals()`, a later caller can reach `type_handler_cache.find()`
  while the losing init thread is still inserting. That is also
  `unordered_map` UB.
- **Note on failed init:** the current source does not provide a strong,
  independent failed-init path for this specific map; the important bug here
  is the unsynchronized first-use init itself.
- **Fix direction:** fix this as part of issue #1 by serializing
  `init_globals()` behind a mutex-protected slow path and publishing success
  with an acquire/release atomic flag. Once init is guaranteed to complete
  before any read, this map becomes effectively immutable for the life of
  the process and is safe to read without locks.

### 5. `init_called` memory ordering (Significant)

- **Shared state:** the `init_called` flag is a plain non-atomic `bool`.
- **Concern:** even if the TOCTOU in issue #1 were resolved by an
  ordinary "check, do init, set flag" idiom, a non-atomic `bool` is still
  a data race under the C++ memory model (concurrent read and write with
  no synchronization). In practice on x86 this is often benign, but a
  reader is not guaranteed to see the preceding writes to
  `type_handler_cache`, `*_str`, `*_attr`, etc. before it sees
  `init_called == true`.
- **Fix direction:** subsumed by the atomic+mutex fix suggested for issue
  #1; once readers use an acquire load and the successful initializer does a
  release store after `init_globals()`, this concern goes away.
- **Listed separately** because if someone "fixes" issue #1 by only adding
  a mutex around `init_globals()` without making the flag read acquire-ordered,
  the memory-ordering hole remains.

### 6. `torch_tensor_cls` first-use capture (Minor)

- **Shared state:** `torch_tensor_cls` is looked up inside
  `init_globals()` only if `torch` is already present in
  `sys.modules`. It is never re-checked.
- **Concern:** this is **not** a real free-threading bug. If the first
  kernel specialization happens before the user imports `torch`,
  `torch.Tensor` is never added to `type_handler_cache`, and later torch
  tensors fall through to the attribute-based fallback at line 557. That
  fallback still specializes tensors correctly; the only effect is that the
  exact-type fast path is missed for the rest of the process.
- **Why no separate issue file / patch:** there is no correctness failure,
  no unsafe shared-state mutation on the hot path after init, and no need to
  change the code for thread-safety. A "fix" that eagerly imports `torch` or
  mutates `type_handler_cache` later would be risky and could regress the
  no-torch dispatch behavior covered by Triton's tests. Keep this only as a
  traceability note in this README.
- **Severity:** Minor and not worth reporting under the `CLAUDE.md`
  guidance.

## Rejected / low-value observations

- **`make_tensordesc_args` and `visit_make_tensordesc_args`** — operate
  only on caller-supplied Python containers (`result` list, `arg`,
  `sig`, `relevant_paths`, `tensordesc_meta`). No file-scope shared
  mutable state. The `PyList_Append` / `PyList_SetSlice` calls act on a
  freshly-created local list; the borrowed reference from
  `PyList_GetItem(tensordesc_meta, ...)` is only safe if the caller
  doesn't mutate `tensordesc_meta` from another thread, but that is a
  caller contract, not a bug in this file.
- **Interned string / attribute globals (`i32_str`, `dtype_attr`, …)** —
  once written by `init_interned_strings()` they are only read. Their
  concurrency story is entirely subsumed by the `init_globals()` TOCTOU
  in issue #1. No separate issue.
- **Class globals (`constexpr_cls`, `tensor_descriptor_cls`, …)** — same
  story: subsumed by issue #1.
- **`align_kwarg`** — initialized once in `init_interned_strings()`; same
  story.
- **Handlers that don't touch file-scope state** (`handle_bool_type`,
  `handle_float_type`, `handle_constexpr_type`, `handle_jit_callable`,
  `handle_long_type`, `handle_tuple`) — read only `from_borrowed_ref` on
  the already-initialized interned strings or recurse back into
  `specialize_arg`. No new shared state.
- **`PyModule_AddFunctions` in `init_native_specialize`** — runs under the
  import lock at module init. Not a race.
- **Lifetime of `PyObject*` values stored in the caches** — all stored
  pointers come from new references and are held for the lifetime of the
  process (no `Py_DECREF` anywhere in this file for the cached values).
  That is a deliberate leak, not a lifetime bug. Worth keeping in mind
  when designing a fix so the fix doesn't accidentally introduce
  use-after-free on eviction.

## Open questions for the deeper audit

- Does any existing caller serialize the first call to
  `native_specialize_impl` (e.g., through some JIT-level lock in
  `runtime/jit.py`) that would make the init TOCTOU unreachable in
  practice? If so, severity of issue #1 may be tier-dependent rather
  than "SEVERE" in the absolute sense. The hot-path cache races
  (issues #2–#4) remain SEVERE either way.
- Are these caches reachable from any `py::gil_scoped_release` region?
  Spot check says no — this file does not declare any GIL-release
  scopes — but confirm during the deeper audit.
- Confirm that `Py_hash_t` collisions between unrelated dtype objects
  can't produce a wrong-type specialization (that is a correctness
  question, independent of threading, but worth a note).
