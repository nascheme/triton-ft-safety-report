# `get_shared_view` / `get_distributed_view` `const_cast` away const on `LinearLayout`

- **Status:** Open
- **Severity:** Significant
- **Component:** `python/src/linear_layout.cc` (lines 177-186)

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

  The `const_cast` strongly suggests `getSharedLayoutStr` /
  `getDistributedLayoutStr` mutate the layout (e.g. lazy normalization,
  caching, sorting bases). Whether they actually do is in
  `lib/Dialect/TritonGPU/IR/Dialect.cpp` and needs confirmation.
- **Writer(s):** Both view bindings, on the receiver `self`.
- **Reader(s):** Any other binding on the same `LinearLayout` (`__eq__`,
  `bases`, `out_dims`, `apply`, `compose`, `invert*`, etc.), and another
  concurrent call to either view binding.
- **Race scenario:** Two threads invoke `layout.get_distributed_view(...)`
  on the same Python `LinearLayout`. Both enter `getDistributedLayoutStr`
  via `const_cast` mutable references and race on whatever internal state
  that helper updates. If the helper is in fact const-correct under the
  hood, this is benign and the binding should drop the cast.
- **Suggested fix:** First, audit `getSharedLayoutStr` and
  `getDistributedLayoutStr` to determine whether they actually mutate the
  passed `LinearLayout`. If they do not, change their signatures to
  `const LinearLayout &` and remove the `const_cast`. If they do mutate
  (e.g. memoize a derived layout), serialize the mutation with an internal
  lock or restructure to compute the derived state without modifying the
  input.
