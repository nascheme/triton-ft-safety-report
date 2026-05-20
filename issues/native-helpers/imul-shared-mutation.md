# `LinearLayout.__imul__` mutates a shared Python object in place

- **Status:** Open
- **Severity:** Significant
- **Component:** `python/src/linear_layout.cc` (lines 165-170)

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
- **Suggested fix:** Make `__imul__` non-mutating from Python's perspective:
  return a fresh `LinearLayout` (`return lhs * rhs;`) so the Python `__imul__`
  protocol rebinds rather than mutating shared C++ state. Alternatively,
  document that `LinearLayout` is not safe to share across threads once any
  thread may call `*=`.
