# `dtype2str` `std::unordered_map` mutated on the tensordesc specialization path

- **Status:** Open
- **Patch:** `dtype2str-unordered-map-race.patch` (guard `dtype2str`
  with a mutex and use a two-phase miss path so
  `canonicalize_dtype_fn` runs outside the critical section)
- **Severity:** SEVERE
- **Component:** `python/src/specialize.cc`
- **Tier:** 2

- **Shared state:** `static Dtype2Str dtype2str` (`specialize.cc:74`), a
  process-global `std::unordered_map<Py_hash_t, PyObject *>`. There is no
  mutex or other synchronization, and free-threaded CPython no longer
  serializes calls into this helper.
- **Writer(s):** `specialize_tensordesc()` cache-miss path at
  `specialize.cc:188`: `dtype2str[dsk] = res.ptr();`. Insertion may rehash,
  invalidating buckets and iterators for any concurrent reader.
- **Reader(s):** `specialize_tensordesc()` lookup at `specialize.cc:180`:
  `auto it = dtype2str.find(dsk);`. `specialize_tensordesc` is invoked from
  the tensordesc handlers registered for `tensor_descriptor_cls`,
  `nvidia_tensor_descriptor_cls`,
  `nvidia_tensor_descriptor_im2col_cls`, and `amd_tensor_descriptor_cls`
  in `init_type_handler_cache()`, and is therefore reached from
  `specialize_arg()` for **every tensordesc / Gluon descriptor argument**
  to **every kernel launch**. The entry point is `native_specialize_impl`,
  called from `runtime/jit.py` per-argument during specialization on the
  hot dispatch path.
- **Race scenario:** Two threads launch Triton kernels with tensordesc
  arguments concurrently. Thread A takes the miss branch with a new
  `dtype_hash` key and is in the middle of `dtype2str[dsk] = res.ptr()`
  (which may rehash the table) while Thread B concurrently calls
  `dtype2str.find(dsk)` from another `specialize_tensordesc()`
  invocation, or takes its own miss branch and inserts. Concurrent
  `unordered_map` insert/find or insert/insert is undefined behavior:
  possible bucket corruption, lost insert, infinite loop in the hash
  table, returning a dangling `PyObject *` from a stale bucket, or a
  hard crash. A returned dangling pointer is then dereferenced via
  `PyObject_Str(type_str)` at `specialize.cc:205` and propagates the
  corruption into the Python caller.

  Even without rehashing, the read of the `PyObject *` value via
  `it->second` races with the bucket's value being written by another
  thread's insert for the same key — there is no happens-before edge
  between writer and reader.
- **Suggested fix:** Guard `dtype2str` with a `std::mutex`, but do not hold
  that mutex while calling `canonicalize_dtype_fn`. Use a two-phase miss
  path instead: lock and check, drop the lock before computing the
  canonicalized dtype, then re-lock, re-check, and publish the new
  `PyObject *` only if another thread has not already inserted it. The
  cached values are intentionally immortal, so the winning thread can
  transfer ownership into the map and the losing thread can simply drop its
  temporary result.
