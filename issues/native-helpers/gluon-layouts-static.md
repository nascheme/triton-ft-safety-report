# `layoutToGluon` static `GluonLayouts` cache

- **Status:** Open
- **Severity:** Minor
- **Component:** `python/src/gluon_ir.cc` (lines 124-178, 201-202)

- **Shared state:** A function-local static `GluonLayouts layouts;` inside
  `layoutToGluon`. The struct holds raw `py::handle` pointers (released
  Python references) to layout classes imported from
  `triton.experimental.gluon.language._layouts`,
  `triton.experimental.gluon.language.amd._layouts`,
  `triton.experimental.gluon.language.nvidia.blackwell`, and
  `triton.experimental.gluon.language.amd.gfx1250._layouts`.
- **Writer(s):** The struct's constructor, run exactly once on the first
  caller of `layoutToGluon` (C++11 magic-static guard).
- **Reader(s):** All subsequent invocations of `layoutToGluon`, called from
  every binding that converts an MLIR layout back to a Python layout
  (`to_linear_layout`, `get_gluon_layout_from_tensor`,
  `get_gluon_layout_from_memdesc`, `compute_tmem_reg_layout`,
  `get_amd_mfma_scale_layout`, `get_amd_wmma_scale_layout`, `create_tmem_load`).
- **Race scenario:** Magic-static initialization is thread-safe, so the
  first-caller race is fenced. Two residual concerns:
  1. The lambda runs Python C API (module imports, attribute lookups) under
     the magic-static lock; under free-threading other threads reaching
     `layoutToGluon` will block on init. No corruption, but a soft
     serialization point on first use.
  2. The handles are stored as raw `py::handle` (post-`.release()`), so the
     `PyObject*` is leaked and lives for the process. Concurrent reads of
     these pointers are safe; concurrent calls into the underlying Python
     types depend on those types' own free-threading safety.
- **Suggested fix:** Likely no change needed. If desired, eagerly construct
  `GluonLayouts` during module init (under the import lock) instead of on
  first call, so the lazy-init serialization point is paid up-front.
