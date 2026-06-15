# `type_handler_cache` init-time race through unsynchronized lazy init

- **Issue-Id:** FT045
- **Status:** Open
- **Rank:** Critical Blocker
- **Component:** `python/src/specialize.cc`
- **Patch:** [`init-globals-toctou.patch`](init-globals-toctou.patch)

- **Shared state:** `static TypeHandlerCache type_handler_cache` in
  `specialize.cc`, a process-global
  `std::unordered_map<PyTypeObject *, TypeHandler>` with no mutex. The map is
  populated during first-use initialization and then treated as read-only.
  `std::unordered_map` is not safe for concurrent insert/insert or
  insert/find; a rehash during another thread's access can corrupt buckets,
  lose entries, return a dangling iterator, or crash outright.
- **Writer(s):** `init_type_handler_cache()` performs
  `type_handler_cache[key] = fn` inserts covering `PyLong_Type`,
  `PyBool_Type`, `PyFloat_Type`, `PyTuple_Type`, and the Triton class
  pointers (`torch.Tensor`, `TensorDescriptor`, Gluon descriptors,
  `constexpr`, `JITCallable`). It is called only from `init_globals()`.
- **Reader(s):** `specialize_arg()`:
  `auto it = type_handler_cache.find(arg_type);`. `specialize_arg()` is
  reached from `specialize_impl`, the `native_specialize_impl`
  METH_FASTCALL exposed to Python and called on the per-argument
  specialization hot path from `runtime/jit.py`.
- **Race scenario:** This is a Critical Blocker consequence of issue FT044's
  unsynchronized lazy init. Two threads can both observe the plain
  `init_called` guard as false and both enter `init_globals()`, which in
  turn calls `init_type_handler_cache()`. Both threads then issue
  `type_handler_cache[...] = ...` inserts simultaneously. That is
  concurrent mutation of the same `std::unordered_map`, which is undefined
  behavior. A later thread can also start reading the map via
  `specialize_arg()` after one init thread has published success while the
  other racing init thread is still inserting.
- **Consequence:** corruption of the process-global type-dispatch table used
  by every specialization call. Plausible outcomes are crashes or other
  undefined behavior inside `unordered_map` operations on the hot path.
- **Suggested fix:** Fix the root cause through issue FT044's atomic+mutex
  `ensure_init()` guard. Once initialization is serialized and success is
  published with release/acquire ordering, `type_handler_cache` is
  effectively immutable for the life of the process and safe to read
  without a separate map lock.
