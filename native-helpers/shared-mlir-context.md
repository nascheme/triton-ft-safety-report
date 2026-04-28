# `linear_layout.cc` shares one process-wide MLIRContext across all calls

- **Status:** Open
- **Severity:** SEVERE
- **Component:** `python/src/linear_layout.cc`

- **Shared state:** A single `MLIRContext` object stored as a leaked
  `PyObject*` in a function-local static (`getLinearLayoutContext`,
  lines 19-28). The context is constructed in `ir.cc` line 324 with
  `MLIRContext::Threading::DISABLED`, so its `StorageUniquer` (the interner
  backing `StringAttr::get`, `IntegerAttr::get`, `RankedTensorType::get`, …)
  takes no internal locks.
- **Writer(s):** Every binding that calls `mlir::StringAttr::get(ctx, …)` on
  this shared context: `LinearLayout.identity_1d`, `LinearLayout.strided_1d`,
  `LinearLayout.zeros_1d`, `LinearLayout.from_bases`, `LinearLayout.apply`,
  plus the `LinearLayout(...)` constructors invoked inside those bindings.
  Each interns one or more attributes by mutating uniquer hash maps.
- **Reader(s):** Same set of bindings, plus any later equality / lookup
  comparisons against previously interned attributes (`__eq__`, `__mul__`,
  `__imul__`, `compose`, `invert_and_compose`, `get_shared_view`,
  `get_distributed_view`, `get_matrix_view`).
- **Race scenario:** Thread A calls `LinearLayout.identity_1d(64, "x", "y")`
  while Thread B calls `LinearLayout.strided_1d(32, 2, "x", "y")`. Both call
  `mlir::StringAttr::get(ctx, …)` concurrently on the same uniquer. The
  uniquer's hash map is mutated by both threads with no locking, leading to
  classic non-thread-safe `unordered_map` corruption: torn writes, lost
  insertions, returning attributes that alias different storage, or crashes
  in subsequent lookups.
- **Suggested fix:** Two main options.
  1. Construct the linear-layout context with
     `MLIRContext::Threading::ENABLED` so the uniquer takes its internal
     locks. Confirm that other Triton code paths do not rely on the context
     remaining single-threaded for ordering.
  2. Wrap every entry into `getLinearLayoutContext()` with a process-wide
     `std::mutex` covering the call into MLIR. Coarser but safe.

  Option 1 is preferred unless there is a reason to keep threading disabled.
  Either way, audit every binding in `linear_layout.cc` (and any caller of
  `getLinearLayoutContext`) against the chosen fix.
