# Gluon IR bindings race on a shared, threading-disabled `MLIRContext`

- **Status:** Open
- **Severity:** Significant
- **Component:** `python/src/gluon_ir.cc`

- **Shared state:** The `MLIRContext` referenced by every `GluonOpBuilder`.
  `GluonOpBuilder`'s base `TritonOpBuilder` holds an `OpBuilder` constructed
  from a non-owning `MLIRContext *` passed in by the Python wrapper
  (`py::init<MLIRContext *>()`, line 326). That context is constructed in
  `ir.cc` line 324 with `MLIRContext::Threading::DISABLED`, so its
  storage uniquer is not internally locked.
- **Writer(s):** Most binding lambdas in this file. Each calls
  `mlir::StringAttr::get(ctx, …)`, `IntegerAttr::get`,
  `RankedTensorType::get`, or one of the `*EncodingAttr::get` factories on
  the same context. Examples: `buildCgaLayoutAttr` (line 36),
  `getCgaLayoutBases` (line 48), `get_blocked_layout`,
  `get_distributed_linear_layout`, `get_padded_shared_layout`,
  `get_shared_linear_layout`, `get_amd_wmma_layout`, `to_linear_layout`,
  `create_local_gather`, `create_local_scatter`, `create_warp_pipeline_border`.
- **Reader(s):** All of the above, plus any binding that compares or looks
  up previously interned attributes / types via `==`, `dyn_cast`, etc.
- **Race scenario:** Two Python threads each hold a `GluonOpBuilder` for the
  same compilation context (or the same builder is shared). Thread A calls
  `builder.get_blocked_layout(...)` while Thread B calls
  `builder.get_distributed_linear_layout(...)`. Both invoke uniquer
  insertions on the same `StorageUniquer` hash map without locking,
  producing classic non-thread-safe-container corruption.
- **Suggested fix:** This is fundamentally a `MLIRContext` issue rooted in
  `ir.cc`. Options:
  1. Construct the per-compilation context with
     `MLIRContext::Threading::ENABLED` so the uniquer takes its own locks.
  2. Document that a single `MLIRContext` (and any builders referencing it)
     must not be used from multiple Python threads, and audit Triton's
     compile path to ensure each thread gets its own context.

  The standalone module helpers in this file
  (`compute_tmem_reg_layout`, `make_cga_layout`, `get_amd_mfma_scale_layout`,
  `get_amd_wmma_scale_layout`, `get_layout_view`) construct a fresh
  stack-local `MLIRContext` and are already safe; they are a useful pattern
  to compare against.
