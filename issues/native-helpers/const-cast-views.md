# `get_shared_view` / `get_distributed_view` `const_cast` away const on `LinearLayout`

- **Issue-Id:** FT031
- **Status:** Not a FT issue — const-correctness fix only
- **Severity:** -
- **Component:** `python/src/linear_layout.cc`
- **Tier:** -
- **Patch:** [`const-cast-views.patch`](const-cast-views.patch)

- **Shared state:** A `LinearLayout` instance owned by a Python wrapper.
  The bindings cast away `const`:

  ```cpp
  .def("get_shared_view",
       [](const LinearLayout &self, bool useHWPointOfView) {
         return mlir::triton::gpu::getSharedLayoutStr(
             const_cast<LinearLayout &>(self), useHWPointOfView);
       })
  .def("get_distributed_view",
       [](const LinearLayout &self, bool useHWPointOfView) {
         return mlir::triton::gpu::getDistributedLayoutStr(
             const_cast<LinearLayout &>(self), useHWPointOfView);
       })
  ```

  The `const_cast` is a gratuitous workaround: both helpers in
  `lib/Dialect/TritonGPU/IR/Dialect.cpp` only call `const` accessors
  (`getInDimNames`, `getOutDimNames`, `getOutDimSizes`, `getInDimSize`,
  `getTotalOutDimSize`, `getNumOutDims`, `apply`) and do not mutate the
  layout. The non-const parameter type is the only reason the cast exists.
- **Writer(s):** Both view bindings, on the receiver `self`.
- **Reader(s):** Any other binding on the same `LinearLayout` (`__eq__`,
  `bases`, `out_dims`, `apply`, `compose`, `invert*`, etc.), and another
  concurrent call to either view binding.
- **Race scenario:** Two threads invoke `layout.get_distributed_view(...)`
  on the same Python `LinearLayout`. Both enter `getDistributedLayoutStr`
  through `const_cast` mutable references. Since the helpers are in fact
  const-correct, the accesses are read-only and benign — but the aliased
  mutable references defeat the type system's read-only guarantee and the
  cast should be dropped.
- **Suggested fix:** Change both helper signatures to `const LinearLayout &`
  and remove the `const_cast` at the binding sites. The compiler then
  enforces read-only access (any future non-const use would fail to
  compile). No locking is needed since neither helper mutates the layout.

---

**Note:** This is **not** a free-threading fix. Both helpers call only
`const` `LinearLayout` methods and do not mutate the object. The
`const_cast` is a const-correctness oversight, not a thread-safety hazard.
This patch should be submitted separately from the FT patch series as a
standalone cleanup.
