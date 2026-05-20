# `dtype_ptr2str` `std::unordered_map` mutated on the hot specialization path

- **Status:** Open
- **Patch:** `dtype-ptr2str-unordered-map-race.patch` (add a
  `std::mutex dtype_ptr2str_mutex`; lookup under the lock, drop it
  before calling `canonicalize_ptr_dtype_fn`, then re-lock to
  `emplace` — loser `Py_DECREF`s its result)
- **Severity:** SEVERE
- **Component:** `python/src/specialize.cc`
- **Tier:** 2

- **Shared state:** `static DtypePtr2Str dtype_ptr2str`
  (`specialize.cc:73`), a process-global
  `std::unordered_map<std::pair<Py_hash_t, bool>, PyObject *, …>`. There is no
  synchronization protecting it. Under free-threaded CPython, attached thread
  state does not serialize callers, so concurrent specialization calls can hit
  this map at the same time.
- **Writer(s):** `handle_tensor()` cache-miss path at `specialize.cc:333`:
  `dtype_ptr2str[dsk] = canon_res;`. Insertion may rehash the table.
- **Reader(s):** `handle_tensor()` lookup at `specialize.cc:322`:
  `auto it = dtype_ptr2str.find(dsk);`. `handle_tensor()` is reached from
  `specialize_arg()` on the `native_specialize_impl` hot path for tensor-like
  arguments, either via the cached `torch.Tensor` type handler or via the
  fallback `data_ptr` attribute check.
- **Race scenario:** Two threads specialize tensor arguments concurrently.
  Thread A takes the miss path for `(dtype_hash, is_const)` and inserts into
  `dtype_ptr2str` while Thread B concurrently does `find()` or performs its
  own insert. Concurrent `std::unordered_map` insert/find or insert/insert is
  undefined behavior: rehash can invalidate buckets observed by the other
  thread, corrupt the container, lose an insert, or crash the process. This is
  on the per-argument specialization path used during kernel dispatch.
- **Suggested fix:** Protect all access to `dtype_ptr2str` with a mutex, but
  keep `canonicalize_ptr_dtype_fn` — a Python call — outside the critical
  section. Two-phase miss path: lock and lookup, drop the lock, call Python,
  re-lock and `emplace`. On a race, the loser's `py::object` is decref'd on
  scope exit and the winner's entry is observed via the `emplace` result.
  Holding a native mutex across an arbitrary Python call risks deadlock if the
  callee re-enters specialization or otherwise tries to take the same mutex.

  Apply the same synchronization strategy to `dtype2str` (issue #3); both
  caches have the same threading hazard.
