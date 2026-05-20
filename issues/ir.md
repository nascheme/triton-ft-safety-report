# python/src/ir.cc Free-Threading Issues

## Issues

| # | Severity | Tier | Component | Issue |
|---|----------|------|-----------|-------|
| 1 | SEVERE      | 1/2 | `PassManager::run` — `::llvm::DebugFlag` / `setCurrentDebugTypes` | Process-global LLVM debug flag and debug-type registry mutated inside a `py::gil_scoped_release` binding body |
| 2 | SEVERE      | 2   | `PassManager::run` — `context->disableMultithreading()`           | Concurrent mutation of a shared `mlir::MLIRContext` with the GIL released |
| 3 | SEVERE      | 2   | `load_dialects` binding                                           | Concurrent `appendDialectRegistry` + `loadAllAvailableDialects` on a shared `MLIRContext` |
| 4 | SEVERE      | 2   | Operation-building bindings (`TritonOpBuilder`, `OpBuilder`, `IntegerType::get`, `StringAttr::get`, …) | Implicit mutation of `MLIRContext`'s type/attribute uniquing tables from multiple threads once the GIL is gone |
| 5 | Significant | 1   | `plugin::loadPlugins()` TOCTOU (transitive via `load_dialects` and init) | `static bool pluginsLoaded` + `static std::vector<TritonPlugin> plugins` guarded only by a plain bool (in `lib/Tools/PluginUtils.cpp`, reachable from `ir.cc`) |
| 6 | Significant | 1/3 | `PassManager::run` — `enableCrashReproducerGeneration` / `enableTiming` / `enableIRPrinting` / diagnostic-handler install | Stateful mutation of `self` and of the shared `MLIRContext` (diagnostic / printing / reproducer state) per call, GIL-released |
| 7 | Significant | 3   | `setupTritonDiagnosticHandler` — `llvm::errs()` sink                | All diagnostic output goes to the process-wide `llvm::errs()`; concurrent emissions interleave |
| 8 | Minor       | 3   | `py_getenv` / `py_getenv_bool` / `get_cache_invalidating_env_vars` | POSIX `getenv` is not thread-safe against `setenv`; these helpers are reachable from any thread |

Issues in `python/src/ir.cc` — the pybind11 surface that wraps
`mlir::MLIRContext`, MLIR/Triton operation builders, and the MLIR
`PassManager`. This is the companion file to `llvm.cc` in pass 3 of the
audit plan: `PassManager::run` is the main GIL-release site, and the
dialect-registration and op-building bindings effectively decide whether an
`MLIRContext` can be shared across threads at all.

Individual issue files should be written for at least #1, #2, #3, #4, and #5.
Items #6–#8 can stay as triage notes unless deeper investigation promotes
them.

See the triage notes below for the reasoning behind each entry and for items
that were considered and rejected.


## Context

### Public pybind11 bindings created by `init_triton_ir`

Organized by what they do with shared state:

- **Context creation / configuration**
  - `py::class_<MLIRContext>(...)` constructor (line 322): builds a context
    with `MLIRContext::Threading::DISABLED`, i.e. MLIR's own internal thread
    pool is off.
  - `printOpOnDiagnostic` / `printStackTraceOnDiagnostic` setters.
  - `load_dialects(context)` (line 336): calls `plugin::loadPlugins()`,
    builds a `DialectRegistry`, `appendDialectRegistry`, and
    `loadAllAvailableDialects`. Directly mutates the context's dialect state.
- **Op / attribute / type construction** (bulk of the file)
  - `py::class_<Type>`, `FunctionType`, `Location`, `Value`, `ModuleOp`,
    `FuncOp`, `OpBuilder`, `TritonOpBuilder`. All of these construct MLIR
    types, attributes, and ops by calling into `MLIRContext` (via
    `IntegerType::get(ctx, ...)`, `StringAttr::get(ctx, ...)`,
    `RoundingModeAttr::get(ctx, ...)`, etc.).
- **Plugin-provided op builders** — registered at module init (line 1863)
  from `plugin::loadPlugins()`.
- **Parsing**
  - `parse_mlir_module(filename, context)` at line 776 — parses an MLIR file
    into the given context.
- **`PassManager`**
  - `pass_manager.enable_debug` (line 1876).
  - `pass_manager.get_pipeline_str`.
  - `pass_manager.run(mod, repro_pipeline_tag)` declared with
    `py::call_guard<py::gil_scoped_release>()` at line 1979. This is the
    single most important GIL-release site in the file.
- **Environment helpers** (via METH_FASTCALL `py_getenv`, `py_getenv_bool`,
  plus `get_cache_invalidating_env_vars`).

### Process-global / shared mutable state touched from these bindings

- `::llvm::DebugFlag` (line 1955, 1964): process-wide LLVM debug toggle.
- `setCurrentDebugTypes(...)` (line 1966): process-wide LLVM debug-type
  registry.
- `llvm::errs()` as the diagnostic sink for every context built via this
  file (`TritonSourceMgrDiagnosticHandler` passes `llvm::errs()` to the
  `SourceMgrDiagnosticHandler` base).
- `plugin::loadPlugins()` — `lib/Tools/PluginUtils.cpp` lines 146–150:
  ```
  static std::vector<TritonPlugin> plugins;
  static bool pluginsLoaded = false;
  const std::vector<TritonPlugin> &loadPlugins() {
    if (pluginsLoaded) return plugins;
    // ... populate plugins ...
  }
  ```
  Plain-bool TOCTOU identical in shape to the `specialize.cc`
  `init_called` issue.
- The Python-exposed `MLIRContext` instance itself. Python objects are
  shared across Python threads; under free-threading two threads can both
  call `load_dialects(ctx)` or both drive a `PassManager::run` that
  references the same context.
- `getenv` / `setenv` (POSIX) — not thread-safe.

### Relevant CLAUDE.md patterns

- Pattern 6 / 9 / 10 (GIL-released regions): `PassManager::run` runs with
  `py::call_guard<py::gil_scoped_release>`. Everything the body touches —
  `llvm::DebugFlag`, `setCurrentDebugTypes`, context mutation, and the pass
  pipeline itself — is concurrently accessible.
- Pattern 1 (lazy native init): `loadPlugins()` is the textbook example,
  transitively reachable from `load_dialects`.


## Triage notes

### 1. `::llvm::DebugFlag` and `setCurrentDebugTypes` mutation with GIL released (SEVERE)

- **Shared state:** `::llvm::DebugFlag` (a plain non-atomic `bool` in LLVM)
  and the LLVM debug-type registry managed by `setCurrentDebugTypes`.
- **Writer:** `PassManager::run` body (lines 1954–1967):
  ```
  if (triton::tools::getBoolEnv("TRITON_ENABLE_LLVM_DEBUG")) {
    ::llvm::DebugFlag = true;
  }
  if (auto debugOnly = triton::tools::getStrEnv("TRITON_LLVM_DEBUG_ONLY");
      !debugOnly.empty()) {
    ::llvm::DebugFlag = true;
    setCurrentDebugTypes(debugTypes.data(), debugTypes.size());
  }
  ```
  These are unconditionally written on every `run()` call when the relevant
  env vars are set.
- **Reader:** the LLVM debug-macros machinery reads `DebugFlag` and the
  current debug-type set from everywhere in LLVM, including from inside the
  MLIR/LLVM pass pipeline that immediately follows.
- **Runs with GIL released** (`py::call_guard<py::gil_scoped_release>()`,
  line 1979).
- **Race scenario:** Thread A and Thread B both enter `PassManager::run`
  with `TRITON_LLVM_DEBUG_ONLY` set. Both write to `::llvm::DebugFlag` and
  both call `setCurrentDebugTypes` with their own `debugTypes` pointer
  (pointing at Thread-A-local storage). LLVM holds onto pointer data from
  `setCurrentDebugTypes` internally; if Thread A's `storage` SmallVector
  goes out of scope while Thread B's pipeline is still running, reads of
  the debug-type list are use-after-free. Even without the UAF, both threads
  race on a plain-`bool` flag and on LLVM's internal debug-type container.
- **Severity:** SEVERE — plain non-atomic bool mutation under concurrency,
  plus potential use-after-free on the debug-type storage.
- **Tier:** Tier 1 / 2 whenever LLVM debug env vars are set. These are
  developer flags but they are not guarded by a debug build.
- **Fix direction:** move the LLVM debug flag setup out of the per-call
  body into a one-shot setup (e.g. at `init_targets` or context creation),
  or guard the mutation with a dedicated mutex. Also fix the
  `setCurrentDebugTypes` lifetime problem by keeping the backing storage
  alive beyond the scope of `run()`. Needs deeper investigation.

### 2. `context->disableMultithreading()` from `PassManager::run` (SEVERE)

- **Shared state:** the Python-exposed `mlir::MLIRContext` passed to
  `PassManager::run` via `mod.getContext()`.
- **Writer:** `PassManager::run` body can call
  `context->disableMultithreading()` at lines 1929 and 1947.
- **Reader:** every pass in the pipeline, plus any other Python thread
  sharing the same context (for op construction or for its own `run()`).
- **Race scenario:** Thread A starts `PassManager::run(modA, ...)` and
  flips `disableMultithreading`. Thread B is simultaneously building ops
  on the same context, or is about to start its own `PassManager::run`.
  The threading-state flip is visible mid-pipeline and is not synchronized.
  Worse, `MLIRContext` itself is not documented as safe for concurrent
  writer + readers across Python threads.
- **Runs with GIL released**.
- **Severity:** SEVERE — concurrent mutation of a non-thread-safe context
  on the pass-manager hot path.
- **Tier:** Tier 2 whenever two Triton compiles share an `MLIRContext`. See
  the open-question at the bottom of this README.
- **Fix direction:** at minimum, do not mutate `disableMultithreading`
  inside `PassManager::run`; set it once at context creation (the context
  is already built with `Threading::DISABLED`, so most of these calls look
  redundant). The broader question — can one `MLIRContext` be driven from
  two threads at all — is issue #3/#4. Needs deeper investigation.

### 3. `load_dialects` on a shared `MLIRContext` (SEVERE)

- **Shared state:** the Python-wrapped `MLIRContext`'s dialect registry
  and its loaded-dialect map.
- **Writer:** `load_dialects(context)` (line 336): builds a
  `DialectRegistry`, appends it to the context, and then calls
  `context.loadAllAvailableDialects()`.
- **Race scenario:** Two Python threads call `load_dialects(ctx)` on the
  same context, or one thread calls `load_dialects(ctx)` while another is
  building ops on `ctx`. `appendDialectRegistry` and `loadAllAvailableDialects`
  mutate per-context data structures that are not thread-safe under
  concurrent readers.
- **Runs with GIL held** (no explicit release). Under free-threaded CPython
  that still means there is no global serialization — the binding executes
  concurrently with other bindings on the same context.
- **Severity:** SEVERE if the Python side ever shares a context across
  threads for dialect loading. If `load_dialects` is always called once per
  context before any concurrency, severity drops to Significant. Needs
  Python-side confirmation.
- **Fix direction:** document the single-threaded-per-context contract and
  enforce it at the Python layer, or guard the mutation with a context-scoped
  mutex.

### 4. MLIR op / attribute / type construction on a shared context (SEVERE)

- **Shared state:** `MLIRContext`'s type-uniquing and attribute-uniquing
  tables. Calls like `IntegerType::get(ctx, 32)`, `StringAttr::get(ctx,
  name)`, `RoundingModeAttr::get(ctx, ...)`, `DenseIntElementsAttr::get(...)`,
  and every op-builder construction reach into these tables.
- **Writers/Readers:** essentially every operation-building binding in this
  file — `TritonOpBuilder`, `OpBuilder`, the op construction helpers on
  `ModuleOp`, `FuncOp`, etc. Line numbers for representative calls include
  397, 478, 772, 825, 895, 900, 1075, 1082, 1089, 1199, 1808, 1815, 1834,
  1840.
- **Race scenario:** Two Python threads drive separate kernels whose
  Python-side code holds (or ends up sharing) the same `MLIRContext`. Each
  emits `IntegerType::get(ctx, 32)` etc. The context's uniquing table is
  mutated concurrently. MLIR does provide some internal synchronization for
  these tables when `Threading` is enabled, but the Triton context is
  explicitly constructed with `Threading::DISABLED` (line 324), which is the
  mode intended for single-threaded use. Under that mode, the uniquing
  tables are **not** expected to be hit concurrently.
- **Severity:** SEVERE if any Python-side sharing of the context exists
  across threads (the most natural way this gets triggered is via shared
  builder / module objects). Even module-local op construction (walk,
  rewrite) races through the shared context if the context is shared.
- **Tier:** Tier 2 or Tier 1 depending on how the Python side structures
  compiles.
- **Fix direction:** either (a) confirm and enforce one `MLIRContext` per
  thread, or (b) rebuild the context with `Threading::ENABLED` so MLIR's
  own locks cover the uniquing tables (this interacts with issue #2 — the
  code explicitly wants `Threading::DISABLED` for diagnostic reasons).
  Needs deeper investigation.

### 5. `plugin::loadPlugins()` plain-bool TOCTOU (Significant)

- **Shared state:** `static std::vector<TritonPlugin> plugins;` and
  `static bool pluginsLoaded = false;` at `lib/Tools/PluginUtils.cpp`
  lines 146–147.
- **Writer:** `loadPlugins()` itself, populating `plugins` and setting
  `pluginsLoaded = true` on success.
- **Reader:** any caller of `loadPlugins()`. In `ir.cc`:
  - line 340 — inside the `load_dialects` binding body. This is a
    **post-init / user-driven** call site: two threads calling
    `load_dialects` could both observe `pluginsLoaded == false` and both
    populate.
  - line 1863 — at module init under Python's import lock. That call site
    is not the problem, but it does not pre-populate `plugins` unless some
    plugin path is set.
- **Race scenario:** Two Python threads call `load_dialects(ctx)` before
  either has triggered a successful `loadPlugins` run. Both observe
  `pluginsLoaded == false` and both `push_back` into `plugins`, and both
  assign `pluginsLoaded = true`. Concurrent `std::vector::push_back` on the
  same vector is undefined behavior.
- **Severity:** Significant — shape is identical to `specialize.cc` issue
  #1 (`init_globals` TOCTOU), but the hot-path reachability is narrower
  (only `load_dialects`, which callers often run once).
- **Fix direction:** use `std::call_once` around the body, or convert the
  flag to `std::atomic<bool>` with acquire/release and guard the slow path
  with a mutex. Note: this fix lives in `lib/Tools/PluginUtils.cpp`, not
  `ir.cc` — but the Triton audit should track it because the only callers
  are `ir.cc`.

### 6. PassManager-wide mutation during `run()` (Significant)

- **Shared state:** the `PassManager self` and the context's
  crash-reproducer / timing / printing configuration.
- **Writer:** `PassManager::run` body calls, on every invocation:
  - `self.enableCrashReproducerGeneration(...)` (lines 1948 or 1951),
  - `self.enableTiming()` (line 1971),
  - `TritonSourceMgrDiagnosticHandler diagHandler =
    setupTritonDiagnosticHandler(context);` which in turn calls
    `context->printOpOnDiagnostic(...)`, `context->printStackTraceOnDiagnostic(...)`,
    and sometimes `context->disableMultithreading()` (line 136).
- **Race scenario:** Two threads running `PassManager::run` with different
  env-var settings can interleave these mutations. In particular the
  `printOpOnDiagnostic` / `printStackTraceOnDiagnostic` state on the
  context is flipped per call and is shared between contexts that two
  threads both hold. `makeReproducer` also writes a file at the configured
  path; if two threads use the same `TRITON_REPRODUCER_PATH` they race on
  the file.
- **Severity:** Significant. Most of these are diagnostic knobs rather
  than codegen-affecting state, so the practical consequence is interleaved
  or corrupted diagnostic output, not wrong-kernel execution. Listed
  separately from #2 because the mutations are conceptually per-call even
  though they alias shared state.
- **Fix direction:** move one-shot context configuration out of the
  per-call body; keep only strictly per-run settings on the `PassManager`
  itself. Needs deeper investigation.

### 7. `setupTritonDiagnosticHandler` — `llvm::errs()` sink (Significant)

- **Shared state:** the process-wide `llvm::errs()` stream.
- **Writer:** every `TritonSourceMgrDiagnosticHandler` constructed in this
  file (including the one built per `PassManager::run` invocation at line
  1974 and the one built by `getTensorDescMetadata` at line 166) installs
  itself with `llvm::errs()` as the backing `raw_ostream`.
- **Race scenario:** Two threads running `PassManager::run` with
  overlapping diagnostics both write to `llvm::errs()`. MLIR's
  `SourceMgrDiagnosticHandler` serializes per-handler, but `llvm::errs()`
  is a singleton and concurrent writes from independent handlers interleave.
- **Severity:** Significant. The consequence is interleaved error output,
  not memory corruption (LLVM's `raw_fd_ostream` is flush-per-write for
  `errs`). But under free-threading the output is no longer deterministic
  and may be hard to debug.
- **Fix direction:** route per-run diagnostics to a per-thread stream and
  emit a copy to `errs` only after the pipeline finishes. Lower priority.

### 8. `py_getenv` / `py_getenv_bool` / `get_cache_invalidating_env_vars` (Minor)

- **Shared state:** the process environment.
- **Writer/Reader:** these helpers call POSIX `getenv`. Triton does not
  `setenv` anywhere in its own code, but the user might.
- **Severity:** Minor. POSIX `getenv` is not thread-safe against `setenv`.
  If the user mutates env vars from another thread, readers can see torn
  strings or a freed buffer. This is a user-contract issue more than a
  Triton bug; list it only to track the surface.
- **Fix direction:** none in Triton; documented caveat at most.


## Rejected / low-value observations

- **`MLIRContext::Threading::DISABLED`** is a deliberate configuration
  choice for diagnostic reasons; it is not itself a bug. It is, however,
  the reason issues #2 and #4 bite: MLIR's own locks are not engaged. So
  the fix direction interacts with this choice but the choice is not the
  bug.
- **`parse_mlir_module`** — if two threads parse into the same context,
  this is subsumed by issue #4 (shared-context uniquing tables).
- **`get_pipeline_str`** — read-only on the PassManager, no shared state
  concerns beyond the PassManager object itself.
- **`PaddingOption` / `CacheModifier` / `MemSemantic` / `MemSyncScope` /
  `EvictionPolicy` / `RMWOp` / `DescriptorReduceKind` / `RoundingMode` /
  `ScaleDotElemType` enum registrations** — ordinary pybind11 enum
  registration at module-init time; no runtime shared-state mutation.
- **`py::enum_<PaddingOption>(m, ..., py::module_local())`, `py::class_<...>
  (..., py::module_local())`, `py::dynamic_attr()` annotations** — purely
  binding-shape concerns; unrelated to free-threading.
- **`py::init<>` constructors for `OpBuilder`, `TritonOpBuilder`,
  `PassManager`** — these construct per-instance objects. The shared state
  they carry (an `MLIRContext*` pointer) is covered under issue #4.
- **`ReproducerStreamFactory makeConsoleReproducer`** — returns a factory
  lambda; per-invocation stream instance. Not shared.
- **`parseCommaSeparatedValues`** — pure function on caller-supplied
  storage. No shared state.
- **`TritonSourceMgrDiagnosticHandler` local `llvm::SourceMgr sourceMgr`
  member** — per-handler instance, not shared between handlers. Its output
  sink is `llvm::errs()` (issue #7), which is shared, but the SourceMgr
  itself is fine.
- **Diagnostic handler returned by value** from
  `setupTritonDiagnosticHandler` — this is a subtle lifetime issue inside
  `PassManager::run` (the handler is stack-local and its destructor runs at
  end of `run()`). Not a free-threading bug; listed here only to note that
  any reuse of the handler address across threads would be wrong.


## Open questions for the deeper audit

- **Does Triton's Python layer ever share a single `MLIRContext` across
  threads?** This is the single most important question for issues #2,
  #3, #4, #6. Start from `triton/python/triton/compiler/compiler.py` and
  the backend-specific compile entry points. If the answer is "each
  compile creates its own context", severity of #2–#4 drops significantly.
  If the answer is "yes, a module-global context is reused", several of
  these issues move from Tier 2 theoretical to Tier 1 immediate.
- **Is there an existing process-wide compile lock?** If `_async_compile.py`
  or the JIT dispatch layer already serializes `PassManager::run`, issues
  #1, #2, #6 become unreachable in practice (but still deserve defensive
  fixes if the lock is ever loosened).
- **Is `::llvm::DebugFlag` documented or observed to be accessed under a
  lock anywhere in Triton?** If not, the mutation in `PassManager::run` is
  a straightforward race; if yes, see whether the lock covers both writer
  and readers.
- **Confirm that `setCurrentDebugTypes` stores the caller's pointer rather
  than copying.** If it stores a copy (in some LLVM versions), the UAF
  concern under issue #1 goes away; the plain-bool race on `DebugFlag`
  remains.
- **Confirm `plugin::loadPlugins()` callers.** The module-init call at
  line 1863 is under the import lock and safe; the `load_dialects` call at
  line 340 is the one that actually creates the TOCTOU. Are there any
  other callers in the wider codebase?
