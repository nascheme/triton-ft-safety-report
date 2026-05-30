# `LinearLayout.__imul__` mutates a shared Python object in place

- **Issue-Id:** FT034
- **Status:** Rejected
- **Severity:** -
- **Component:** `python/src/linear_layout.cc`
- **Tier:** -
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

---

**Rejected.** `LinearLayout` is a pure value type with no static state,
no shared caches, and no lazy singletons. In practice, instances are
created during layout conversion and then consumed locally within a single
compilation context — there is no code path that stores a `LinearLayout`
into a shared cache or registry. Sharing a single `LinearLayout` instance
across threads requires intentional misuse that violates the intended usage
pattern. The proposed fix also introduces an unacceptable API bifurcation:
`a *= b` mutates on GIL builds but rebinds on free-threaded builds, which
would silently break code that relies on in-place semantics. Since the race
scenario is not reachable in practice, this is not a meaningful
free-threading issue and the patch should be dropped.
