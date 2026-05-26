# python/src/ir.cc Free-Threading Issues

## Issues

| # | Severity | Tier | Component | Issue |
|---|----------|------|-----------|-------|
| 1 | LOW       | 2   | `PassManager::run` — `::llvm::DebugFlag` / `setCurrentDebugTypes` | Process-global LLVM debug flag and debug-type registry mutated inside a `py::gil_scoped_release` binding body. Debug/diagnostic-only; does not affect codegen. |
| 2 | LOW       | 2   | `PassManager::run` — `context->disableMultithreading()`           | Latent only: per-compile context, not shared across threads in the shipping codebase |
| 3 | LOW       | 2   | `load_dialects` binding                                           | Latent only: per-compile context, not shared across threads in the shipping codebase |
| 4 | LOW       | 2   | Operation-building bindings (`TritonOpBuilder`, `OpBuilder`, `IntegerType::get`, `StringAttr::get`, …) | Latent only: per-compile context, not shared across threads in the shipping codebase |
| 5 | LOW       | 1   | `plugin::loadPlugins()` TOCTOU (transitive via `load_dialects` and init) | `static bool pluginsLoaded` + `static std::vector<TritonPlugin> plugins` guarded only by a plain bool (in `lib/Tools/PluginUtils.cpp`). Pattern is unsound but not currently reachable: `init_triton_ir` and `init_plugin_passes` both call `loadPlugins()` at module init under the import lock, so `pluginsLoaded == true` by the time any user thread can call `load_dialects`. Latent risk only. |
| 6 | LOW       | 2   | `PassManager::run` — `enableCrashReproducerGeneration` / `enableTiming` / `enableIRPrinting` / diagnostic-handler install | Per-call mutations on `self` and `mod.getContext()`. Backends create `PassManager` and `MLIRContext` fresh per compile, so these mutate per-compile objects, not shared state. Only residual race is the reproducer file write when `TRITON_REPRODUCER_PATH` is set — debug-only. |
| 7 | MED | 3   | `setupTritonDiagnosticHandler` — `llvm::errs()` sink                | All diagnostic output goes to the process-wide `llvm::errs()`; concurrent emissions interleave |
| 8 | LOW       | 3   | `py_getenv` / `py_getenv_bool` / `get_cache_invalidating_env_vars` | POSIX `getenv` is not thread-safe against `setenv`; these helpers are reachable from any thread |

Issues in `python/src/ir.cc` — the pybind11 surface that wraps
`mlir::MLIRContext`, MLIR/Triton operation builders, and the MLIR
`PassManager`. This is the companion file to `llvm.cc` in pass 3 of the
audit plan: `PassManager::run` is the main GIL-release site, and the
dialect-registration and op-building bindings effectively decide whether an
`MLIRContext` can be shared across threads at all.

No individual issue files are warranted at this time. The Tier 2 HIGH
candidates #2/#3/#4 were downgraded to LOW after verifying that
`mlir::MLIRContext` is constructed fresh per `compile()` call and is
never shared across Python threads in the shipping codebase. The unsafe
patterns remain real and would become live races if a future refactor
ever cached or pooled contexts; see the open-questions resolution at the
bottom for the trace.

See the triage notes below for the reasoning behind each entry and for items
that were considered and rejected.


## Context

### Public pybind11 bindings created by `init_triton_ir`

Organized by what they do with shared state:

- **Context creation / configuration**
  - `py::class_<MLIRContext>(...)` constructor: builds a context with
    `MLIRContext::Threading::DISABLED`, i.e. MLIR's own internal thread
    pool is off.
  - `printOpOnDiagnostic` / `printStackTraceOnDiagnostic` setters.
  - `load_dialects(context)`: calls `plugin::loadPlugins()`, builds a
    `DialectRegistry`, `appendDialectRegistry`, and
    `loadAllAvailableDialects`. Directly mutates the context's dialect state.
- **Op / attribute / type construction** (bulk of the file)
  - `py::class_<Type>`, `FunctionType`, `Location`, `Value`, `ModuleOp`,
    `FuncOp`, `OpBuilder`, `TritonOpBuilder`. All of these construct MLIR
    types, attributes, and ops by calling into `MLIRContext` (via
    `IntegerType::get(ctx, ...)`, `StringAttr::get(ctx, ...)`,
    `RoundingModeAttr::get(ctx, ...)`, etc.).
- **Plugin-provided op builders** — registered at module init from
  `plugin::loadPlugins()`.
- **Parsing**
  - `parse_mlir_module(filename, context)` — parses an MLIR file
    into the given context.
- **`PassManager`**
  - `pass_manager.enable_debug`.
  - `pass_manager.get_pipeline_str`.
  - `pass_manager.run(mod, repro_pipeline_tag)` declared with
    `py::call_guard<py::gil_scoped_release>()`. This is the single
    most important GIL-release site in the file.
- **Environment helpers** (via METH_FASTCALL `py_getenv`, `py_getenv_bool`,
  plus `get_cache_invalidating_env_vars`).

### Process-global / shared mutable state touched from these bindings

- `::llvm::DebugFlag`: process-wide LLVM debug toggle, mutated inside
  `PassManager::run`.
- `setCurrentDebugTypes(...)`: process-wide LLVM debug-type registry,
  also called inside `PassManager::run`.
- `llvm::errs()` as the diagnostic sink for every context built via this
  file (`TritonSourceMgrDiagnosticHandler` passes `llvm::errs()` to the
  `SourceMgrDiagnosticHandler` base).
- `plugin::loadPlugins()` in `lib/Tools/PluginUtils.cpp`:
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

### 1. `::llvm::DebugFlag` and `setCurrentDebugTypes` mutation with GIL released (LOW)

- **Shared state:** `::llvm::DebugFlag` (a plain non-atomic `bool` in LLVM)
  and the LLVM debug-type registry managed by `setCurrentDebugTypes`.
- **Writer:** `PassManager::run` body:
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
- **Runs with GIL released** (`py::call_guard<py::gil_scoped_release>()`
  on the `pass_manager.run` binding).
- **Race scenario:** Thread A and Thread B both enter `PassManager::run`
  with `TRITON_LLVM_DEBUG_ONLY` set. Both write to `::llvm::DebugFlag` and
  both call `setCurrentDebugTypes`. The plain-`bool` write is technically a
  data race; concurrent `setCurrentDebugTypes` calls also race on LLVM's
  internal debug-type container (upstream LLVM copies the strings into a
  `std::vector<std::string>`, so there is no UAF on the caller's storage,
  but the vector itself is mutated unlocked).
- **Severity:** LOW.
  - The flags only gate `LLVM_DEBUG(...)` stderr output; they do not affect
    codegen, cache state, or kernel correctness. This matches the
    "benign debug/print flag staleness" exclusion in `CLAUDE.md`.
  - Both threads read the same env var and write the same value, so the
    steady-state outcome is identical; transient races only affect what
    debug lines print.
  - Triggered only when developer env vars (`TRITON_ENABLE_LLVM_DEBUG`,
    `TRITON_LLVM_DEBUG_ONLY`) are set.
- **Tier:** Tier 2 (only races when two threads compile concurrently). This
  is essentially a debug-feature analogue of the LLVM `cl::opt` mutation
  case the README puts under Tier 3, but it's not user-driven knob mutation
  so it's listed here rather than promoted.
- **Fix direction:** move the LLVM debug flag setup out of the per-call
  body into a one-shot setup (e.g. at `init_targets` or context creation),
  or guard the mutation with a dedicated mutex. Low priority given the
  diagnostic-only impact.

### 2. `context->disableMultithreading()` from `PassManager::run` (LOW — latent only)

- **Shared state (candidate):** the Python-exposed `mlir::MLIRContext`
  passed to `PassManager::run` via `mod.getContext()`.
- **Writer:** `PassManager::run` body can call
  `context->disableMultithreading()` (from both the crash-reproducer
  setup and the diagnostic-handler setup).
- **Why not currently a bug:** every `ir.context()` caller in the
  shipping codebase creates the context as a local variable and uses it
  within a single compile call (see the open-questions resolution at the
  bottom). Two concurrent compiles each get their own `MLIRContext`, so
  the per-call mutation operates on a per-compile object, not shared
  state. Also note that `MLIRContext` is constructed with
  `Threading::DISABLED` already (in `ir.cc`), so the per-call
  `disableMultithreading()` is a same-value write even on the owning
  thread.
- **Severity:** LOW — the pattern of mutating context state inside a
  GIL-released binding body is unsound and would be a HIGH race the
  moment any path caches or pools `MLIRContext` objects, but it is not
  reachable in the current code.
- **Tier:** Tier 2 in shape, but unreachable.
- **Fix direction:** drop the redundant `disableMultithreading()` calls
  from `PassManager::run` — the context is already constructed in that
  mode. Low priority defensively.

### 3. `load_dialects` on a shared `MLIRContext` (LOW — latent only)

- **Shared state (candidate):** the Python-wrapped `MLIRContext`'s
  dialect registry and its loaded-dialect map.
- **Writer:** `load_dialects(context)`: builds a `DialectRegistry`,
  appends it to the context, and then calls
  `context.loadAllAvailableDialects()`.
- **Why not currently a bug:** `load_dialects` is called exactly once
  per context in the shipping codebase, immediately after the fresh
  `ir.context()` construction and before any other thread can see the
  context. Call sites:
  - `compiler.py:301` — right after `context = ir.context()` at the top
    of `compile()`, on a function-local context.
  - `IRSource.__init__` (`compiler.py:95`) — invoked from
    `compile()` at line 240 on the same function-local context.
  - `_filecheck.py:71` — function-local context.
  - `test_bindings.py:95` — function-local context.
  - `proton/hooks/instrumentation.py:236` — function-local context.

  No path shares the resulting context with another thread, so the
  per-context mutations cannot race.
- **Severity:** LOW — the binding does not release the GIL itself, but
  the GIL is no longer a serializer under free-threading. If a future
  change ever exposes a context for cross-thread reuse, this becomes a
  HIGH race on the dialect registry. Worth a watcher.
- **Fix direction:** if shared contexts are ever introduced, either
  enforce single-threaded `load_dialects` per context (e.g. eager load
  at context construction) or guard the mutation with a context-scoped
  mutex. No fix needed today.

### 4. MLIR op / attribute / type construction on a shared context (LOW — latent only)

- **Shared state (candidate):** `MLIRContext`'s type-uniquing and
  attribute-uniquing tables. Calls like `IntegerType::get(ctx, 32)`,
  `StringAttr::get(ctx, name)`, `RoundingModeAttr::get(ctx, ...)`,
  `DenseIntElementsAttr::get(...)`, and every op-builder construction
  reach into these tables. The context is built with
  `Threading::DISABLED`, so MLIR's own per-table locks are not engaged.
- **Writers/Readers:** every operation-building binding in this file —
  `TritonOpBuilder`, `OpBuilder`, op construction helpers on `ModuleOp`,
  `FuncOp`, etc.
- **Why not currently a bug:** the context is per-compile (see the
  open-questions resolution at the bottom). All op/attribute/type
  construction inside a `compile()` call runs on that call's private
  context. The compiled-kernel result (`CompiledKernel`) retains a
  reference to the source IR module (and transitively the context),
  but no further op construction happens against that retained context
  after `compile()` returns — it is effectively frozen. Two concurrent
  compiles operate on disjoint contexts and disjoint uniquing tables.
- **Severity:** LOW — the unsafe pattern (uniquing-table mutation
  with no lock) is real, but unreachable while contexts are
  per-compile. The day a context cache or pool is introduced, this
  becomes a HIGH race against MLIR's internal containers and against
  borrowed type/attribute pointers.
- **Fix direction:** when shared contexts are introduced, either pin
  one context per thread or rebuild the context with
  `Threading::ENABLED` so MLIR's locks cover the uniquing tables (this
  interacts with the explicit `Threading::DISABLED` choice made for
  diagnostic reasons). No fix needed today.

### 5. `plugin::loadPlugins()` plain-bool TOCTOU (LOW — latent only)

- **Shared state:** `static std::vector<TritonPlugin> plugins;` and
  `static bool pluginsLoaded = false;` in `lib/Tools/PluginUtils.cpp`.
- **Writer:** `loadPlugins()` itself, populating `plugins` (only if
  `TRITON_PLUGIN_PATHS` is set) and unconditionally setting
  `pluginsLoaded = true` at the end.
- **Reader:** any caller of `loadPlugins()`. Call sites:
  - `init_triton_ir` at `ir.cc:1863` — module init, under the import lock.
  - `init_plugin_passes` at `passes.cc:107` — module init, under the
    import lock.
  - `load_dialects` binding at `ir.cc:340` — post-init / per-compile.
- **Why not currently a bug:** the two module-init call sites both run
  `loadPlugins()` before user Python code can run. `loadPlugins()` sets
  `pluginsLoaded = true` unconditionally on first call (even when the env
  var is unset). So by the time any thread reaches the `load_dialects`
  binding, `pluginsLoaded == true` and every caller takes the
  `if (pluginsLoaded) return plugins;` fast path. The race window is
  closed.
- **Severity:** LOW — the TOCTOU pattern is unsound, but unreachable in
  the current binding layout. Worth tracking as a latent risk only: if
  either module-init call to `loadPlugins()` is removed in a refactor,
  the race becomes live (two compile threads racing through
  `load_dialects` would both observe `pluginsLoaded == false` and both
  `push_back` into `plugins`).
- **Fix direction (defensive):** use `std::call_once` around the body, or
  convert the flag to `std::atomic<bool>` with acquire/release and guard
  the slow path with a mutex. Fix lives in `lib/Tools/PluginUtils.cpp`.

### 6. PassManager-wide mutation during `run()` (LOW)

- **Shared state (candidate):** the `PassManager self` and the
  `mod.getContext()` configuration (crash-reproducer / timing /
  printing / diagnostic state).
- **Writer:** `PassManager::run` body calls, on every invocation:
  - `self.enableCrashReproducerGeneration(...)`,
  - `self.enableTiming()`,
  - `TritonSourceMgrDiagnosticHandler diagHandler =
    setupTritonDiagnosticHandler(context);` which in turn calls
    `context->printOpOnDiagnostic(...)`, `context->printStackTraceOnDiagnostic(...)`,
    and sometimes `context->disableMultithreading()`.
- **Why not actually a race in practice:**
  - Backend compilers construct a fresh `PassManager` per stage
    (e.g. `pm = ir.pass_manager(mod.context)` in
    `third_party/nvidia/backend/compiler.py` and equivalents). The
    PassManager is not shared across threads.
  - `compiler.py` constructs a fresh `MLIRContext` per compile
    (`context = ir.context()`). The context mutated here is the
    per-compile context, not a process-global one.
  - So the per-call mutations modify per-compile objects, not shared
    state. No cross-thread race on the PassManager or context fields
    themselves.
- **Residual concern (LOW):** `makeReproducer(..., reproducerPath)`
  writes a file when `TRITON_REPRODUCER_PATH` is set. The path is
  suffixed with `repro_pipeline_tag`, so two concurrent compiles at the
  same pipeline stage with the same env var collide on the same file.
  Debug/diagnostic-only — does not affect codegen.
- **Tier:** Tier 2. (The original "Tier 1" label was wrong: Tier 1 has
  only one Triton thread so per-call mutations cannot race.)
- **Fix direction:** if the reproducer file race ever matters, include a
  pid/thread-id in the path. Lower priority.

### 7. `setupTritonDiagnosticHandler` — `llvm::errs()` sink (MED)

- **Shared state:** the process-wide `llvm::errs()` stream.
- **Writer:** every `TritonSourceMgrDiagnosticHandler` constructed in this
  file (including the one built per `PassManager::run` invocation and the
  one built by `getTensorDescMetadata`) installs itself with `llvm::errs()`
  as the backing `raw_ostream`.
- **Race scenario:** Two threads running `PassManager::run` with
  overlapping diagnostics both write to `llvm::errs()`. MLIR's
  `SourceMgrDiagnosticHandler` serializes per-handler, but `llvm::errs()`
  is a singleton and concurrent writes from independent handlers interleave.
- **Severity:** MED. The consequence is interleaved error output,
  not memory corruption (LLVM's `raw_fd_ostream` is flush-per-write for
  `errs`). But under free-threading the output is no longer deterministic
  and may be hard to debug.
- **Fix direction:** route per-run diagnostics to a per-thread stream and
  emit a copy to `errs` only after the pipeline finishes. Lower priority.

### 8. `py_getenv` / `py_getenv_bool` / `get_cache_invalidating_env_vars` (LOW)

- **Shared state:** the process environment.
- **Writer/Reader:** these helpers call POSIX `getenv`. Triton does not
  `setenv` anywhere in its own code, but the user might.
- **Severity:** LOW. POSIX `getenv` is not thread-safe against `setenv`.
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
  threads? — Resolved: NO.** Every `ir.context()` caller constructs the
  context as a function-local variable and consumes it within that
  call. Traced call sites:
  - `triton/python/triton/compiler/compiler.py:239` — `compile()`,
    IRSource path. Local; `IRSource` is built one line later and the
    `assert isinstance(src, str)` rules out smuggling in a shared
    `IRSource`.
  - `triton/python/triton/compiler/compiler.py:300` — `compile()`,
    ASTSource path. Local.
  - `triton/python/triton/_filecheck.py:70` — test helper, local.
  - `triton/python/test/unit/runtime/test_bindings.py:91` — test,
    local.
  - `triton/python/test/unit/tools/test_irsource.py:43`/`:88` — tests,
    local.
  - `triton/third_party/proton/proton/hooks/instrumentation.py:235` —
    `init_handle`, local to that call.

  No module-global context exists; no backend stores a context (the
  backend compilers reach the context only via `mod.context`, which was
  attached by the per-compile path); `CompiledKernel` retains the source
  module (and transitively the context), but no further op
  construction, dialect loading, or `PassManager::run` runs against
  that retained context after `compile()` returns.

  Consequence: issues #2, #3, #4 were downgraded from HIGH to LOW
  (latent only). They are real patterns of unsafe context mutation that
  would fire immediately if any path ever cached or pooled contexts.
- **Is there an existing process-wide compile lock?** If `_async_compile.py`
  or the JIT dispatch layer already serializes `PassManager::run`, issues
  #1, #2, #6 become unreachable in practice (but still deserve defensive
  fixes if the lock is ever loosened).
- **Is `::llvm::DebugFlag` documented or observed to be accessed under a
  lock anywhere in Triton?** If not, the mutation in `PassManager::run` is
  a straightforward race; if yes, see whether the lock covers both writer
  and readers.
- **Confirm that `setCurrentDebugTypes` stores the caller's pointer rather
  than copying.** Upstream LLVM appears to copy into a
  `std::vector<std::string>`, which is why issue #1 is currently LOW (no
  UAF, just a diagnostic-only race). Worth confirming for the LLVM revision
  Triton vendors.
- **Confirm `plugin::loadPlugins()` callers.** Currently two module-init
  call sites (`init_triton_ir`, `init_plugin_passes`) under the import
  lock pre-warm `pluginsLoaded = true`, defusing the post-init
  `load_dialects` TOCTOU. If a refactor removes either of those calls,
  the race becomes live. Worth a watcher.
