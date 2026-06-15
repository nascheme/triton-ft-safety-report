# `dtype_ptr2str` and `dtype2str` `std::unordered_map` caches mutated on the specialization hot path

- **Issue-Id:** FT042
- **Status:** Open
- **Rank:** Critical Blocker
- **Component:** `python/src/specialize.cc`
- **Patch:** [`dtype-ptr2str-unordered-map-race.patch`](dtype-ptr2str-unordered-map-race.patch)

Two sibling caches in `specialize.cc` share the same hazard and the same
fix. They are tracked together here; FT043 covers `dtype2str` and is
resolved by this patch.

- **Shared state:** two process-global `std::unordered_map`s with no
  synchronization:
  - `static DtypePtr2Str dtype_ptr2str` —
    `std::unordered_map<std::pair<Py_hash_t, bool>, PyObject *, …>`, used
    in `handle_tensor()`.
  - `static Dtype2Str dtype2str` —
    `std::unordered_map<Py_hash_t, PyObject *>`, used in
    `specialize_tensordesc()`.
- **Writer(s):** cache-miss path in each function:
  - `handle_tensor()`: `dtype_ptr2str[dsk] = canon_res;` after calling
    `canonicalize_ptr_dtype_fn`. Insertion may rehash.
  - `specialize_tensordesc()`: `dtype2str[dsk] = res.ptr();` after calling
    `canonicalize_dtype_fn`. Insertion may rehash.
- **Reader(s):** the matching `find(dsk)` in the same function. Both
  `handle_tensor()` and `specialize_tensordesc()` are reached from
  `specialize_arg()` on the `native_specialize_impl` hot path, per
  argument, on every kernel launch — `handle_tensor` for tensor-like
  args, `specialize_tensordesc` for tensordesc / Gluon descriptor args.
- **Race scenario:** Two threads launch kernels concurrently. Thread A
  takes a miss path and is partway through inserting (possibly rehashing)
  while Thread B does `find()` or its own insert on the same map.
  Concurrent `std::unordered_map` insert/find or insert/insert is
  undefined behavior: bucket corruption, lost insert, infinite loop in
  the table, a dangling `PyObject *` returned from a stale bucket, or a
  hard crash. A returned dangling pointer is then dereferenced (e.g.
  `PyObject_Str(type_str)` in `specialize_tensordesc`) and propagates the
  corruption to the Python caller. Even without rehashing, the read of
  the `PyObject *` value via `it->second` races with the bucket's value
  being written by another thread's insert for the same key — there is
  no happens-before edge between writer and reader.
- **Suggested fix:** Give each cache its own `std::mutex` and use a
  two-phase miss path so the Python `canonicalize_*_dtype` call stays
  outside the critical section: lock and look up; drop the lock and call
  Python; re-lock and `emplace`. Holding a native mutex across an
  arbitrary Python call risks deadlock if the callee re-enters
  specialization. The cached `PyObject *` values are intentionally
  immortal, so on a race the losing thread's `py::object` is decref'd on
  scope exit and the winning thread's entry is observed via the
  `emplace` result.

**Insert-only:** Neither map has any call to `erase`, `remove`, or `clear`
anywhere in `specialize.cc`. Stored `PyObject *` values are deliberately
leaked and live for the process lifetime. This insert-only property is what
makes the two-phase miss path safe: an entry found under the lock in phase 1
cannot be invalidated during the unlocked window before phase 2.
