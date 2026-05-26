# `LinearLayout.__imul__` mutates a shared Python object in place

- **Status:** Open
- **Severity:** MED
- **Component:** `python/src/linear_layout.cc`
- **Tier:** 2
- **Patch:** [`imul-shared-mutation.patch`](imul-shared-mutation.patch)

- **Shared state:** A `LinearLayout` instance held by a Python wrapper. The
  binding declares:

  ```cpp
  .def(
      "__imul__",
      [](LinearLayout &lhs, const LinearLayout &rhs) -> LinearLayout & {
        lhs *= rhs;
        return lhs;
      },
      py::return_value_policy::reference_internal)
  ```

  `LinearLayout` is a value type whose internal storage (`bases`, `outDims`,
  std containers) is mutated in place by `operator*=`.
- **Writer(s):** Any thread calling `layout *= other` on the Python object,
  or `__imul__` directly.
- **Reader(s):** Any concurrent reader: `__mul__`, `__eq__`, `compose`,
  `invert*`, `get_*_names`, `bases`, `out_dims`, `apply`, `__repr__` /
  `__str__`, etc.
- **Race scenario:** Thread A holds Python reference `ll` and calls
  `ll *= other` while Thread B reads `ll.bases` or evaluates `ll == ll2`.
  The C++ `operator*=` performs unsynchronized writes to the same `bases`
  map / vectors that Thread B reads, producing a data race on non-atomic
  C++ containers (use-after-free or torn read possible).
- **Suggested fix:** Guard the `__imul__` binding with
  `#ifndef Py_GIL_DISABLED` so it is only exposed on GIL-enabled builds.
  On free-threaded builds Python's `a *= b` falls back to `a = a * b`
  (via `__mul__`), which rebinds the name instead of mutating shared
  C++ state. This avoids any API change for existing GIL-build users
  who may rely on the in-place semantics.
