# python/src/passes.cc Free-Threading Issues

## Issues

| # | Severity | Tier | Component | Issue |
|---|----------|------|-----------|-------|
| 1 | Minor       | 2   | `ModuleAllocation` / `ModuleMembarAnalysis` bindings | Latent only: would mutate the caller's `MLIRContext`, but the context is per-compile in the shipping codebase (same family / same downgrade as `ir.md` issue #4). Also, the `passes.analysis.allocation` / `passes.analysis.membar` bindings have no Python callers in the shipping tree today; real `ModuleAllocation` / `ModuleMembarAnalysis` use lives entirely inside C++ passes |
| 2 | Minor       | 1   | `init_plugin_passes` → `mlir::triton::plugin::loadPlugins()` (`lib/Tools/PluginUtils.cpp:147–188`) | Plain-`bool` TOCTOU on `static bool pluginsLoaded` + `static std::vector<TritonPlugin> plugins`. Not currently reachable: module-init call sites in `passes.cc` and `ir.cc` (both under the import lock) pre-warm `pluginsLoaded = true`, so post-init `load_dialects` callers always hit the fast path. Latent risk — cross-reference to `ir.md` issue #5 |
| 3 | Minor       | 2   | Pass-builder lambdas registered by `ADD_PASS_WRAPPER_*` / `ADD_PASS_OPTION_WRAPPER_*` | Each lambda does `pm.addPass(builder(...))` on a caller-supplied `mlir::PassManager`. Two threads sharing one `PassManager` race on that object; two threads on separate `PassManager`s do not. Contract concern, not a file-local bug — see triage notes |
| 4 | Minor       | 2   | Pass-builder factory functions (`createTritonGPUCoalesce`, `createConvertTritonToTritonGPU`, …) | `ADD_PASS_WRAPPER_0` etc. invoke MLIR pass-factory free functions each time the binding is called. Most are plain `new PassT(...)`, but a deeper audit should spot-check for factory-local static state (per-pass option registries, one-shot diagnostics, plugin-side factories) |
| 5 | Not worth reporting | — | `m.def(...)` / `py::class_<…>` registrations inside `init_triton_passes_*` | Registrations run once from the `PYBIND11_MODULE` body under CPython's import lock — write-once module init, covered by CLAUDE.md "low-value patterns" |

Issues in `python/src/passes.cc` — the pybind11 surface that exposes MLIR
pass-pipeline builder functions and two analysis wrappers
(`ModuleAllocation`, `ModuleMembarAnalysis`). This is pass-3 item #5 in the
audit plan, below `specialize.cc`, `llvm.cc`, and `ir.cc`.

No individual issue files are warranted. `passes.cc` is structurally very thin
and its substantive concurrency risks route back through `mlir::PassManager` /
`mlir::MLIRContext` semantics (already tracked in `ir.md`) or through the
plugin loader (already tracked in `ir.md` issue #5). Item #1 was downgraded
to Minor after verifying that (a) `MLIRContext` is per-compile in the shipping
codebase — same resolution as `ir.md` issues #2/#3/#4 (see the open-questions
resolution at the bottom of `ir.md`) — and (b) the `passes.analysis.allocation`
/ `passes.analysis.membar` bindings have no Python callers in the shipping
tree, so the analyses are not reached from Python at all today. Items #3–#5
are kept in the table as traceability notes. See the triage notes below for
reasoning and rejected observations.


## Context

### What `passes.cc` actually contains

The file is ~155 lines and, together with `passes.h`, consists almost
entirely of macro-expanded `m.def(...)` registrations that bind MLIR pass
builders into Python submodules. There is no file-scope mutable state, no
static caches, no lazy initializers owned by this file, and no
`py::gil_scoped_release` / `call_guard<gil_scoped_release>` sites.

Top-level entry point:

- `init_triton_passes(m)` — called exactly once from
  `main.cc`'s `PYBIND11_MODULE` body via
  `init_triton_passes(m.def_submodule("passes"))`. Creates the following
  submodules and wires their init functions:
  - `passes.analysis` → `init_triton_analysis`
  - `passes.common` → `init_triton_passes_common`
  - `passes.convert` → `init_triton_passes_convert`
  - `passes.ttir` → `init_triton_passes_ttir`
  - `passes.ttgpuir` → `init_triton_passes_ttgpuir`
  - `passes.llvmir` → `init_triton_passes_llvmir`
  - `passes.gluon` → `init_gluon_passes`
  - `passes.plugin` → `init_plugin_passes`

### What each init function does

- `init_triton_analysis` (`passes.analysis`) — registers two classes:
  - `py::class_<mlir::ModuleAllocation>(m, "allocation", py::module_local())`
    with `py::init<mlir::ModuleOp>()`. Constructs `ModuleAllocation` from
    a caller-supplied `mlir::ModuleOp`.
  - `py::class_<mlir::ModuleMembarAnalysis>(m, "membar", py::module_local())`
    with `py::init<mlir::ModuleAllocation *>()` and
    `.def("run", &mlir::ModuleMembarAnalysis::run)`. The `run` method
    performs membar analysis over the module associated with the allocation
    analysis. Neither binding declares `py::gil_scoped_release`.
- `init_triton_passes_common`, `…_convert`, `…_ttir`, `…_ttgpuir`,
  `…_llvmir`, `init_gluon_passes` — all call `ADD_PASS_WRAPPER_*` /
  `ADD_PASS_OPTION_WRAPPER_*` macros. Each macro expands to
  `m.def(name, [](mlir::PassManager &pm, …){ pm.addPass(factory(…)); })`.
  The `m.def("add_canonicalize_llvm_ir", …)` registration is the only
  hand-written lambda; it uses `pm.addNestedPass<LLVM::LLVMFuncOp>(...)`
  but is otherwise the same shape.
- `init_plugin_passes` — calls
  `mlir::triton::plugin::loadPlugins()` and iterates
  `plugin.listPasses()`, registering one `m.def(pass.name, …)` per pass.
  This is the file's single call into the process-wide plugin registry.

### What shared mutable state is touched

*Not* by this file directly. What is actually shared:

1. The caller-supplied `mlir::PassManager` passed into every registered
   lambda. If two threads share one `PassManager` object they race on its
   internal pass list. `passes.cc` does not own or create `PassManager`
   instances, so this is a caller contract, not a file-scope issue.
2. The `mlir::MLIRContext` implied by the caller's `mlir::ModuleOp` (in
   `init_triton_analysis`) and by the passes eventually run through the
   `PassManager`. Concurrent analysis runs against modules sharing a context
   can mutate type/attribute uniquing tables — the same failure mode as
   `ir/issues.md` issue #4. `passes.cc` is one more path that lets Python
   code reach that state.
3. `mlir::triton::plugin::loadPlugins()` — the process-global plugin vector
   and its `static bool pluginsLoaded` TOCTOU guard. `passes.cc` reaches it
   at module init under the import lock; `ir.cc` can reach it again later
   on the hot path. See `ir/issues.md` issue #5.

### Where the GIL is released

Nowhere in `passes.cc`. All registered lambdas and class methods run with
the calling thread's thread state attached. Under free-threading, that does
**not** prevent other Python threads from running concurrently — it only
means Python C API calls in these bindings are safe for the calling thread.

## Triage notes

### 1. `ModuleAllocation` / `ModuleMembarAnalysis` — MLIRContext concurrency (Minor — latent only)

- **Shared state (candidate):** whatever state inside `mlir::MLIRContext`
  the allocation-analysis construction and `ModuleMembarAnalysis::run`
  transitively touch (type/attribute uniquing tables, op insertion during
  membar synthesis, diagnostic engine, …). The module and its context are
  caller-supplied.
- **Writer(s):** `ModuleMembarAnalysis::run` inserts barrier ops into the
  module; the `ModuleAllocation` constructor performs whole-module
  analysis. Both effectively mutate MLIR-side structures hanging off the
  context.
- **Why not currently a bug:**
  1. The `passes.analysis.allocation` / `passes.analysis.membar` bindings
     have no Python call sites in the shipping tree (grep of
     `python/`, `third_party/` for `passes.analysis`, `analysis.membar`,
     `analysis.allocation` returns no matches). Real
     `ModuleAllocation` / `ModuleMembarAnalysis` use is entirely inside
     C++ passes (`third_party/{nvidia,amd}/lib/.../TritonGPUToLLVM.cpp`
     etc.), reached only via `PassManager::run`, not via these bindings.
  2. Even if the bindings were called, the caller-supplied `MLIRContext`
     is per-compile in the shipping codebase. This is the same downgrade
     argument that took `ir.md` issues #2/#3/#4 from SEVERE to Minor —
     see the "Open questions" resolution at the bottom of `ir.md` for
     the traced call sites (`compiler.py:239`/`:300`, `_filecheck.py:70`,
     proton instrumentation, tests). Two concurrent compiles operate on
     disjoint contexts, so the per-context mutation cannot race.
- **Severity:** Minor — the unsafe pattern (no MLIR-side lock because
  the context is built with `Threading::DISABLED`) is real and would
  fire immediately if (a) these analysis bindings ever gain a Python
  caller *and* (b) a future refactor caches or pools `MLIRContext`
  objects across threads. Today neither condition holds.
- **Fix direction:** nothing to fix in `passes.cc` alone. Resolution is at
  the `MLIRContext` design level (one context per thread, or rebuild with
  `Threading::ENABLED`) and would land alongside the `ir.md` #2/#3/#4
  family. No fix needed today.
- **Tier:** Tier 2 in shape, but unreachable.

### 2. `init_plugin_passes` → `loadPlugins()` TOCTOU (Minor — latent only, cross-reference)

- **Shared state:** `static std::vector<TritonPlugin> plugins` and
  `static bool pluginsLoaded` in `lib/Tools/PluginUtils.cpp`'s `loadPlugins`.
- **Writer(s):** the first caller of `loadPlugins()` after process start.
  Populates `plugins` (only if `TRITON_PLUGIN_PATHS` is set) and
  unconditionally sets `pluginsLoaded = true`.
- **Reader(s):** module-init callers `init_plugin_passes`
  (`passes.cc:107`) and `init_triton_ir` (`ir.cc:1863`), both under the
  Python import lock; plus the post-init `load_dialects` binding at
  `ir.cc:340`.
- **Why not currently a bug:** module-init callers pre-warm
  `pluginsLoaded = true` before user code can run, so every subsequent
  `load_dialects` invocation hits the `if (pluginsLoaded) return plugins;`
  fast path. The TOCTOU is unreachable in the current binding layout.
- **Severity:** Minor — latent risk only. If a refactor removes either
  module-init pre-warm, two concurrent compile threads in `load_dialects`
  would race on `pluginsLoaded` / `push_back(plugins)`.
- **Why not a separate issue file:** bug lives in
  `lib/Tools/PluginUtils.cpp`; cross-referenced from `ir.md` issue #5.

### 3. Pass-builder lambdas as caller-PassManager mutators (Minor)

- **Shared state:** the caller's `mlir::PassManager` argument. `PassManager`
  holds an internal pass list.
- **Writer(s):** every registered lambda (`pm.addPass(...)`,
  `pm.addNestedPass<...>(...)`). Each call mutates the pass list of the
  supplied `PassManager`.
- **Reader(s):** later calls on the same `PassManager` (including its own
  `run`) traverse the pass list.
- **Race scenario:** if two Python threads call
  `triton._C.libtriton.passes.ttgpuir.add_coalesce(pm)` and
  `…add_pipeline(pm, …)` on the **same** `pm`, they concurrently mutate
  `pm`'s internal vector-like pass list — UB. If each thread uses its own
  `PassManager`, there is no shared state.
- **Why not Significant:** Triton's normal flow constructs a fresh
  `PassManager` per compile (see `compiler/` path and `ir.cc`
  `PassManager` bindings) and populates it sequentially before `.run`.
  Sharing a partially-populated `PassManager` across threads would be an
  unusual caller pattern.
- **Fix direction:** this is a caller-contract issue. If Triton's
  higher-level code starts sharing `PassManager` instances across threads
  during pipeline construction, the fix belongs there, not in `passes.cc`.
- **Tier:** Tier 2.

### 4. Pass-builder factory functions — unverified factory-local state (Minor)

- **Shared state:** unknown. Each `ADD_PASS_WRAPPER_*` invocation calls a
  free function like `mlir::createSCCPPass`,
  `mlir::triton::gpu::createTritonGPUCoalesce`,
  `mlir::triton::plugin::TritonPlugin::addPass`, etc. In the common case
  these are `return std::make_unique<PassT>(...)` — pure factories with no
  shared state.
- **Concern:** a minority of MLIR / Triton passes register pass-option
  parsers, plugin-provided callbacks, or one-shot diagnostics when first
  constructed. Those would be shared mutable state reachable from the
  pass-builder lambda.
- **Why Minor:** no specific factory has been confirmed to hold such
  state; this item is a deliberate "did we actually check?" reminder for
  the deeper audit rather than a concrete bug.
- **Fix direction:** grep the factories actually called here
  (`createTriton*`, `createGluon*`, `createConvert*`, `createArithTo…`,
  `createLLVMDIScope`, `createCanonicalizeLLVMIR`) for file-scope static
  state during the deeper pass; promote or drop this item based on what
  turns up.
- **Tier:** Tier 2, contingent on factory implementation.

### 5. `m.def` / `py::class_` registrations under import lock (Not worth reporting)

- Same reasoning as `src-main/issues.md` item #2. The entire body of
  `init_triton_passes` and its sub-init functions runs during
  `PYBIND11_MODULE` execution, under CPython's import lock. Write-once,
  not observable as mid-init state by other threads.

## Rejected / low-value observations

- **`py::module_local()` on `ModuleAllocation` / `ModuleMembarAnalysis`.**
  Scopes pybind11 type registration to this module's type namespace; it
  affects cross-module type visibility, not thread safety.
- **`ADD_PASS_WRAPPER_*` / `ADD_PASS_OPTION_WRAPPER_*` macro expansions
  (`passes.h`).** All expand into `m.def(name, lambda)` calls that
  complete during module init under the import lock. The lambda bodies
  themselves are the item #3 concern above.
- **Captured `pass` in `init_plugin_passes`'s lambda (`[pass]`).**
  The capture copies each `TritonPlugin::Pass` struct into the closure.
  If plugins keep a pointer into their own static state, that pointer is
  shared by all copies — but that is a plugin-side concern, not a
  `passes.cc` concern.
- **`py::arg("pm"), py::arg("args") = std::vector<std::string>()` defaults
  on plugin passes.** Default argument objects are constructed once at
  registration time. Subsequent calls receive fresh copies. Not a shared
  mutable state issue.
- **Lambdas taking `mlir::PassManager &pm` by non-const reference.** This
  is the pybind11-idiomatic way to mutate the caller's object. Not a race
  unless the caller shares the `PassManager` (see item #3).
- **Use of `mlir::PassManager::addPass` / `addNestedPass`.** These are MLIR
  APIs whose thread safety is a property of `PassManager`, not of this
  file.

## Open questions for the deeper audit

- Confirm that `mlir::ModuleAllocation` and `mlir::ModuleMembarAnalysis`
  do not stash pointers in any process-global registry. If they do, the
  "analysis is local to the caller's module" assumption in item #1 is
  wrong and severity rises.
- Confirm that Triton's Python-side pipeline construction always creates a
  fresh `mlir::PassManager` per kernel compile (check `compiler/compiler.py`
  and related). If so, item #3 stays Minor. If pipelines are cached or
  reused, it rises.
- Cross-check the list of pass factories invoked from this file against
  their implementations for hidden static state (item #4). A targeted grep
  for `static` inside `createTriton*` / `createGluon*` /
  `createConvert*` / `createLLVMDIScope` / `createCanonicalizeLLVMIR`
  should be enough.
- Resolved: `init_plugin_passes` (and `init_triton_ir`) both call
  `loadPlugins()` under the import lock and `pluginsLoaded` is set
  unconditionally on first call, so post-init `load_dialects` callers
  always hit the fast path. Issue downgraded to Minor / latent.
