# python/src/llvm.cc Free-Threading Issues

## Issues

| # | Severity | Tier | Component | Issue |
|---|----------|------|-----------|-------|
| 1 | SEVERE      | 1/2 | `setLLVMOption` / `restoreLLVMOption` / `ScopedLLVMOption` | Process-global `llvm::cl::opt<>` state mutated from GIL-released compile paths; RAII "original-value" capture interleaves between threads |
| 2 | SEVERE      | 1/2 | `dumpSchedulingDAG` stderr redirection                     | Process-wide `dup` / `freopen(stderr)` / `dup2` sequence races with other threads and other Python I/O |
| 3 | Significant | 1/2 | `llvm::TimePassesIsEnabled` / `llvm::TimePassesPerRun`     | Process-global timing flags and `reportAndResetTimings` used from GIL-released `translateLLVMIRToASM` |
| 4 | Significant | 1   | `init_targets` — `llvm::parallel::strategy` write          | Unconditional write outside the `std::call_once` block; data race on the global strategy object |
| 5 | Significant | 2   | `to_module` binding                                        | `py::call_guard<py::gil_scoped_release>` around `translateModuleToLLVMIR` with a user-supplied `mlir::ModuleOp` and `llvm::LLVMContext` |
| 6 | Significant | 2   | `optimize_module` binding                                  | `py::call_guard<py::gil_scoped_release>` around `PassBuilder` / `ModulePassManager::run` operating on a user-supplied `llvm::Module` |
| 7 | Significant | 2   | `llvm::Module` mutation bindings (`add_flag`, `add_fn_attr`, `set_nvvm_maxnreg`, `link_extern_libs`, ...) | Free-threaded Python removes the GIL; concurrent mutation of a shared `llvm::Module` / `llvm::Function` is UB |
| 8 | Minor       | 3   | `init_triton_stacktrace_hook` / `llvm::sys::AddSignalHandler` | Process-global signal handler registration; typically one-shot but not guarded |

Issues in `python/src/llvm.cc` — the pybind11 surface that wraps LLVM/MLIR
translation, module-level optimization, and assembly/object emission. This is
the highest-priority native file after `specialize.cc` per
`AUDIT_PLAN.md` because almost every public binding either releases the GIL
explicitly (`translate_to_asm`, `translate_to_mir`, `translate_mir_to_asm`,
`dump_sched_dag`) or declares a `py::call_guard<py::gil_scoped_release>`
(`to_module`, `optimize_module`), and those regions touch process-global LLVM
state.

Individual issue files should be written for at least the SEVERE items
(#1, #2) and likely for the Significant items #3 and #5/#6 as well. Issues
#4, #7, #8 can stay as triage notes unless deeper investigation promotes them.

See the triage notes below for the reasoning behind each entry and for items
that were considered and rejected.


## Context

### Public pybind11 bindings in `init_triton_llvm`

All of the following are reachable on the compile path:

- `to_module(mod, ctx)` — MLIR ModuleOp -> `llvm::Module`. Declared with
  `py::call_guard<py::gil_scoped_release>()` (line 622).
- `attach_datalayout(mod, triple, proc, features)` — GIL held; creates a
  `TargetMachine` just to compute a data layout.
- `optimize_module(mod, opt, arch, features, flags, enable_fp_fusion)` —
  runs the LLVM new pass-manager module pipeline. Declared with
  `py::call_guard<py::gil_scoped_release>()` (line 759).
- `translate_to_asm(llvmIR, triple, proc, features, flags, enable_fp_fusion, isObject)`
  — explicit `py::gil_scoped_release allow_threads;` scope around
  `parseIR` + `translateLLVMIRToASM` (line 769).
- `dump_sched_dag(...)` — explicit `py::gil_scoped_release allow_threads;`
  (line 797).
- `translate_to_mir(...)` — explicit `py::gil_scoped_release allow_threads;`
  (line 821).
- `translate_mir_to_asm(...)` — explicit `py::gil_scoped_release allow_threads;`
  (line 849).
- `init_targets()` — initializes all LLVM targets under `std::call_once`, then
  unconditionally writes `llvm::parallel::strategy`.
- `link_extern_libs(dstMod, paths)` — GIL held; mutates `dstMod` and its
  `LLVMContext`.

Module-like helper bindings (`py::class_<llvm::Module>`,
`py::class_<llvm::Function>`, `py::class_<llvm::Module::FunctionListType>`)
expose mutators like `add_flag`, `add_fn_attr`, `remove_fn_attr`,
`add_fn_asan_attr`, `add_fn_target_feature`, `set_nvvm_maxnreg`,
`set_calling_conv`, and iteration. These all directly modify the underlying
`llvm::Module` / `llvm::Function` and share the caller's `llvm::LLVMContext`.

### Process-global state touched from GIL-released regions

- `llvm::cl::getRegisteredOptions()` — the process-global option registry.
  Every `setLLVMOption` / `restoreLLVMOption` call looks up an entry and calls
  `addOccurrence` / `setValue` on the stored `llvm::cl::opt<T>*`. These are
  **process globals**, not per-module / per-context state.
- `llvm::TimePassesIsEnabled` and `llvm::TimePassesPerRun` — process-global
  flags; also `reportAndResetTimings` resets process-wide counters.
- `llvm::parallel::strategy` — process-global thread-pool strategy.
- `stderr` / `stdout` — process-wide FILE* and file descriptors; manipulated
  in `dumpSchedulingDAG`.
- LLVM signal handler chain (`llvm::sys::AddSignalHandler`).

### Relevant CLAUDE.md patterns

- Pattern 6 ("code running with detached thread state"): every binding above
  that declares `py::call_guard<py::gil_scoped_release>()` or opens an explicit
  `py::gil_scoped_release allow_threads;` runs with the Python thread state
  detached. Any Python C API call inside is unsafe; any shared native state is
  concurrently accessible. `llvm.cc` exercises several such regions.
- Pattern 9 and 10 (pybind11 GIL-release variants): both variants appear in
  this file.
- Pattern 1 ("lazy native initialization"): `init_targets` is close to this
  pattern, though it uses `std::call_once` for the one-shot part.


## Triage notes

### 1. Global LLVM command-line option state (SEVERE)

- **Shared state:** `llvm::cl::getRegisteredOptions()` returns a process-wide
  map of option name → `llvm::cl::Option*`. The underlying `llvm::cl::opt<T>`
  objects are not documented as thread-safe and definitely not intended to be
  flipped at runtime from concurrent callers.
- **Writers:** `setLLVMOption<bool>`, `setLLVMOption<std::string>`,
  `restoreLLVMOption<*>`, and the `ScopedLLVMOption` RAII wrapper. Every
  `setLLVMOption` call performs:
  - a lookup in the registered-options map,
  - a `getValue()` read on the stored `cl::opt<T>*`,
  - a write via `addOccurrence(1, name, value)` or `setValue(value)`.
- **Call sites that write this global state:**
  - `dumpSchedulingDAG` (flags + `ScopedLLVMOption` for `stop-after` and
    `misched-print-dags`) — runs with GIL released.
  - `translateLLVMIRToMIR` (flags + `print-after-all` + `ScopedLLVMOption`
    for `stop-before`) — runs with GIL released.
  - `translateLLVMIRToASM` (flags + `print-after-all`) — runs with GIL
    released.
  - `translateMIRToASM` (`ScopedLLVMOption` for `start-before`,
    `enable-misched`, `enable-post-misched`; flags + `print-after-all`) —
    runs with GIL released.
  - `optimize_module` binding body (flags + `print-after-all`) — declared with
    `py::call_guard<py::gil_scoped_release>()`.
- **Race scenario A — concurrent option mutation:** Thread A enters
  `translate_to_asm` and flips `print-after-all = true`. Thread B enters
  `translate_mir_to_asm` at the same time and flips `start-before =
  machine-scheduler`. Both threads call into `llvm::cl::opt<>`'s
  `addOccurrence` / `setValue` on different option objects, but the registry
  lookup and the option objects' counters are not designed for concurrent
  mutation. At best the second thread's flags land while the first thread is
  mid-pipeline, pushing the first thread's codegen down a completely different
  path; at worst the `cl::opt<>` internals corrupt.
- **Race scenario B — RAII "original-value" capture:** Thread A constructs
  `ScopedLLVMOption<bool> mischedPrintGuard("misched-print-dags", true);`,
  capturing the original value `false`. Thread B then constructs the same
  scope, captures what it sees (`true`, set by Thread A), and when Thread B's
  scope exits it "restores" to `true`, leaving the option permanently enabled
  after both threads finish. Even without a C++ memory-model race, the
  save/restore pairing is logically wrong under concurrency.
- **Severity:** SEVERE. Process-wide mutable state, mutated concurrently with
  no synchronization, from a GIL-released hot compile path. Effect is not just
  a benign inefficiency: options like `stop-before`, `start-before`,
  `enable-misched`, and `DISABLE_LLVM_OPT`-forwarded flags directly change
  which LLVM passes run and therefore affect generated code correctness and
  output.
- **Tier:** Tier 1/2. Triggered whenever two threads compile concurrently.
- **Fix direction:** see the extended discussion below.

#### Extended fix discussion for issue #1

##### Why a per-option or split-lock approach does not work

`llvm::cl::opt<>` is a process-global command-line option registry designed
to be parsed once at startup.  Triton uses it as a runtime per-compile
configuration mechanism (set flags → run pipeline → restore flags), which is
inherently unsafe under concurrency.  There is no LLVM API to redirect option
reads to per-thread or per-pipeline storage: LLVM passes read from the global
`cl::opt<>` objects directly, by name, during both pipeline construction and
(for flags like `print-after-all` and `misched-print-dags`) during pipeline
execution.

Even making individual `addOccurrence` / `setValue` calls atomic would not
fix the issue, because the bug is in the *sequence*:

1. Thread A captures original value, sets option to X.
2. Thread B captures what it sees (X, set by A), sets option to Y.
3. Thread B restores to X (A's intermediate value, not the true original).
4. Thread A restores to the true original.

The save/restore invariant cannot be made correct with per-option locking —
the critical section must span the entire capture-set-run-restore sequence.

##### The GIL does not protect this

The GIL was already released before entering these functions
(`py::gil_scoped_release`).  Two Python threads can both be in the
GIL-released C++ region simultaneously.  The race therefore exists in
GIL-enabled Python too, if thread-based concurrent compilation workers are
used (e.g. `ThreadPoolExecutor`).  Subprocess-based workers are safe because
each process has its own address space and its own copy of the cl::opt
globals.  In practice, typical Triton usage (single compile thread, or
subprocess workers) avoids the race, which is why it has not been reported
yet.  Free-threading makes it a frontline bug because concurrent compilation
from multiple Python threads is a primary use case.

##### Recommended fix: process-wide mutex, active only in free-threaded builds

The minimal correct fix is a single process-wide mutex held for the entire
body of each function that touches `cl::opt<>` state:
`translateLLVMIRToASM`, `translateLLVMIRToMIR`, `translateMIRToASM`,
`dumpSchedulingDAG`, and the `optimize_module` lambda.

Holding the mutex for the full function body (including `ScopedLLVMOption`
construction and destruction) makes the existing save/restore logic correct —
no other thread can touch the options during the window, so the
original-value capture is always accurate.

Guard the mutex behind `#ifdef Py_GIL_DISABLED` to avoid any overhead in
GIL-enabled builds and to signal clearly that this is a free-threading
concern:

```cpp
#ifdef Py_GIL_DISABLED
static std::mutex g_llvm_codegen_mutex;
#endif

// At the top of translateLLVMIRToASM (and each affected function):
#ifdef Py_GIL_DISABLED
std::lock_guard<std::mutex> lk(g_llvm_codegen_mutex);
#endif
```

No GIL-deadlock risk: the GIL is already released before these functions are
entered, so the calling thread never holds the GIL while waiting for
`g_llvm_codegen_mutex`.

##### Why serializing compilation is acceptable at this stage

For typical Triton users (e.g. vllm), what matters is parallel kernel
*execution*, not parallel kernel *compilation*.  Kernels are compiled once on
first use and the result is written to Triton's on-disk cache; subsequent
runs skip compilation entirely.  Serializing `translate_to_asm` under a mutex
means concurrent compile requests queue up, but in practice they are often
waiting for the same kernel anyway.  After the warm-up phase, all threads
execute pre-compiled kernels with no mutex involved.

Parallel compilation across multiple cores remains possible by using
subprocess workers (fork or spawn) rather than threads, since each process
has its own cl::opt state.

##### Longer-term direction (not required for initial free-threading support)

Making `translate_to_asm` and related functions truly safe for concurrent
in-process use would require one of:

1. Migrating the codegen pipeline from LLVM's legacy pass manager to the new
   pass manager, which accepts per-pipeline configuration through
   `PipelineTuningOptions` and instrumentation callbacks rather than global
   cl::opts.  This is a significant multi-month effort and depends on LLVM
   version support.

2. A patch to LLVM itself to provide per-call or per-thread option overrides
   in the codegen path.

Neither is appropriate for an initial, minimal free-threading patch.  A
comment in the code should note the limitation and point to this issue for
follow-up.

### 2. `dumpSchedulingDAG` stderr redirection (SEVERE)

- **Shared state:** the process's `stderr` FILE* and file-descriptor 2.
- **Writer/Reader:** `dumpSchedulingDAG` (lines 203–231) runs:
  - `int saved_stderr_fd = dup(fileno(stderr));`
  - `FILE *redirected = freopen(dumpFilename.c_str(), "a", stderr);`
  - run the codegen pipeline
  - `dup2(saved_stderr_fd, fileno(stderr));`
  - `close(saved_stderr_fd);`
- **Race scenario:** Thread A redirects stderr to `fileA.txt`. Before it can
  restore, Thread B redirects stderr to `fileB.txt`. When either thread
  "restores", it restores to whatever `saved_stderr_fd` it personally captured,
  which may no longer match the process's stderr state. Other Python threads
  writing to `sys.stderr` during this window also end up appending to the dump
  file.
- **Runs with GIL released** (the binding is `dump_sched_dag`, which opens an
  explicit `py::gil_scoped_release allow_threads;` at line 797 before calling
  `dumpSchedulingDAG`).
- **Severity:** SEVERE. Debug/diagnostic path, but still hits real
  process-global state and corrupts the stderr state for the entire process.
  Although only active when `TRITON_DUMP_MIR` is set, the consequence is not
  limited to the thread that enabled it.
- **Tier:** Tier 1/2 whenever `TRITON_DUMP_MIR` is on.
- **Fix direction:** serialize the entire stderr-redirection region under a
  dedicated mutex, or avoid process-wide stderr redirection altogether (e.g.
  capture LLVM diagnostics through an LLVM `raw_ostream` rather than the C
  `stderr` FD). Needs deeper investigation.

### 3. `llvm::TimePassesIsEnabled` / `llvm::TimePassesPerRun` (Significant)

- **Shared state:** two process-global flags plus LLVM's internal
  `TimePassesHandler` counters used by `reportAndResetTimings`.
- **Writer/Reader:** `translateLLVMIRToASM` lines 365–368 set these to true
  when `LLVM_ENABLE_TIMING` is on. Later calls to `reportAndResetTimings`
  (lines 377, 400) flush and reset the global counters.
- **Race scenario:** Thread A sets the flags and produces a timing report.
  Thread B mid-pipeline also calls `reportAndResetTimings`, clearing Thread
  A's in-progress counters and printing interleaved timing output to
  `llvm::dbgs()` (itself a process-global stream).
- **Runs with GIL released** (`translate_to_asm` uses
  `py::gil_scoped_release`).
- **Tier:** Tier 1 when `LLVM_ENABLE_TIMING` is on; timing results become
  incoherent across threads.
- **Severity:** Significant rather than SEVERE because the artifact is
  diagnostic (pass-timing reports) rather than generated code, but the global
  counters and flags are genuinely raced.
- **Fix direction:** either serialize the timing-enabled path under the same
  mutex as issue #1 (logical since both concern process-global compile state),
  or do not mutate the global timing flags from pybind11 bindings at all and
  instead set them through LLVM's existing API once during `init_targets`.

### 4. `init_targets` — `llvm::parallel::strategy` write after `call_once` (Significant)

- **Shared state:** `llvm::parallel::strategy` — a process-global object
  controlling the default parallel strategy.
- **Writer:** `init_targets` (lines 863–877): a `std::call_once` block does
  the one-shot target init correctly, but the following line
  `llvm::parallel::strategy = llvm::hardware_concurrency(1);` is **outside**
  the once block and runs on every call.
- **Race scenario:** Two threads calling `init_targets` concurrently both
  perform assignment to `llvm::parallel::strategy`. Even though the value is
  the same, under the C++ memory model this is a data race on a non-atomic
  object.
- **Severity:** Significant. Same-value race on an ordinary object is UB per
  C++ even if it appears benign on tested platforms. Also, if a future change
  makes the assigned value non-constant (e.g. parameterize concurrency), the
  race becomes materially wrong.
- **Fix direction:** move the `llvm::parallel::strategy` assignment inside
  the `std::call_once` block so the write is serialized with the target
  initialization.

### 5. `to_module` binding runs MLIR -> LLVM IR with GIL released (Significant)

- **Binding:** `m.def("to_module", ...)` at line 612, declared with
  `py::keep_alive<0, 2>()` and
  `py::call_guard<py::gil_scoped_release>()`.
- **Shared state:** the user-supplied `mlir::ModuleOp` and the user-supplied
  `llvm::LLVMContext`. Both are Python-wrapped objects that can be referenced
  from multiple threads. MLIR IR data structures and `llvm::LLVMContext` are
  not thread-safe for concurrent access.
- **Race scenario:** Two Python threads call `to_module(mod_a, ctx)` and
  `to_module(mod_b, ctx)` with the same `llvm::LLVMContext` wrapper. With the
  GIL released, `translateModuleToLLVMIR` runs concurrently, mutating the
  shared context (types, constants, metadata interning tables) without
  synchronization.
- **Tier:** Tier 2 (and Tier 1 if the Python side keeps a single
  `llvm::LLVMContext` and hands it to compiles from multiple threads — which
  is exactly what `triton/python/triton/backends/*.py` and
  `triton/python/triton/compiler/compiler.py` should be audited for).
- **Fix direction:** either document and enforce that each thread uses its
  own `LLVMContext`, or add a context-level lock. Requires confirming the
  Python-side usage pattern before choosing. Needs deeper investigation.

### 6. `optimize_module` binding runs the LLVM pass pipeline with GIL released (Significant)

- **Binding:** `m.def("optimize_module", ...)` at line 642, declared with
  `py::call_guard<py::gil_scoped_release>()` (line 759).
- **Shared state:** the user-supplied `llvm::Module*`. Its `LLVMContext` is
  also shared with anything else that has a handle to it.
- **Race scenario A:** Two threads call `optimize_module` on the same module
  (shouldn't happen in practice but is not prevented at the binding level).
- **Race scenario B:** Two threads call `optimize_module` on different
  modules that share an `LLVMContext`. The context's interning tables and
  metadata uniquing maps are mutated concurrently.
- **Also touches issue #1** via the flags-forwarding block at the top.
- **Severity:** Significant. Depends on the actual Python-side usage; if
  each compile uses its own module and its own context, only the option-globals
  issue (#1) bites. If a context is reused across threads, this becomes a
  real container race in LLVM's internals.
- **Fix direction:** confirm the Python-side model. If contexts are reused,
  either serialize or redesign.

### 7. `llvm::Module` mutation bindings on shared module objects (Significant)

- **Shared state:** the Python-wrapped `llvm::Module*` and
  `llvm::Function*`.
- **Exposed mutators:** `add_flag` (line 558), `add_fn_attr`,
  `remove_fn_attr`, `add_fn_asan_attr`, `add_fn_target_feature`,
  `set_nvvm_maxnreg` (which also calls
  `getOrInsertNamedMetadata().addOperand()` — a module-level mutation), and
  `set_calling_conv`, plus the `FunctionListType` iterator.
- **Race scenario:** Under free-threading there is no implicit GIL
  serialization between Python callers. Two threads holding the same Python
  `llvm::module` wrapper and calling any combination of these mutators race on
  the underlying `llvm::Module`.
- **Tier:** Tier 2 (two threads operating on the same compiled-kernel object).
  Probably not common in current Triton usage — each compile typically owns
  its own module — but should be confirmed from the Python side.
- **Fix direction:** if the Python side actually holds one module per
  compile, document that and stop there. If modules are reused, a module-level
  lock is needed. Needs deeper investigation.

### 8. `init_triton_stacktrace_hook` / `AddSignalHandler` (Minor)

- **Shared state:** LLVM's process-global signal-handler list.
- **Writer:** `init_triton_stacktrace_hook` — called from module init, only
  if `TRITON_ENABLE_PYTHON_STACKTRACE` is on.
- **Concern:** `llvm::sys::AddSignalHandler` mutates a process-wide handler
  chain. Called from Python module init, so ordinarily protected by Python's
  import lock. Listed here only because the function is *callable* from
  Python and a second caller would race.
- **Severity:** Minor. Realistically a one-shot at import time.


## Rejected / low-value observations

- **`py::class_<llvm::LLVMContext>` / `py::class_<llvm::SourceMgr>` / etc.
  constructor bindings.** These just expose `llvm::LLVMContext`'s default
  constructor to Python. No shared state here — the shared-state risk lives
  on the Python side (one context used from many threads), not in the binding.
  Captured under issue #5 instead.
- **`py::module_local()` on exposed classes.** Purely about symbol visibility
  across pybind11 modules; unrelated to free-threading.
- **`py::make_iterator` on `FunctionListType`.** The `keep_alive<0,1>` keeps
  the container alive; does not protect against concurrent mutation of the
  underlying `llvm::Module::FunctionListType`. Subsumed by issue #7.
- **`createTargetMachine` helper.** Creates per-call local state; each caller
  gets its own `TargetMachine`. No shared state.
- **`getBoolEnv` / `getStrEnv` calls inside GIL-released regions.** These
  read the process environment (`std::getenv`). POSIX `getenv` is not
  thread-safe against `setenv`, but Triton does not mutate its own env. If a
  user concurrently mutates their own env from another thread, that is a
  user contract issue, not a Triton-side bug.
- **`link_extern_libs` local `std::unordered_set<std::string> externalFns`.**
  This is function-local, not shared. The real concern in `link_extern_libs`
  is mutation of `dstMod` and its context, which is captured under issue #7.
- **`report_fatal_error` call sites.** These abort the process on error.
  Not a free-threading issue.
- **`std::once_flag init_flag` in `init_targets`.** C++11 magic statics / 
  `std::call_once` are thread-safe by the standard. Only flagged here for
  the *sibling* write at line 876, which is issue #4.
- **`py::keep_alive<0, 2>()` on `to_module`.** This is a pybind11 refcount
  contract to keep the `LLVMContext` alive for the returned `llvm::Module`.
  Unrelated to free-threading — but it does confirm that the context is
  logically shared between Python and the returned module, which is the
  setup for issue #5.


## Open questions for the deeper audit

- Does the Python side ever call any of the GIL-released bindings concurrently
  from multiple threads with a shared `llvm::LLVMContext`? Start from
  `triton/python/triton/compiler/compiler.py` and the backend-specific
  compile paths (NVIDIA/AMD backends). This determines whether issues #5
  and #6 are Tier 2 theoretical or Tier 1/2 immediate.
- Is there already a process-wide compile lock in Python (e.g. in
  `compiler.py` or `_async_compile.py`) that would implicitly serialize these
  calls? If so, the severity of #1–#3 drops until that lock is removed.
- Does `llvm::cl::opt<>` have any LLVM-side documentation or test coverage
  for concurrent `addOccurrence` / `setValue`? A quick pass through LLVM's
  own usage would confirm whether a single mutex over the entire compile body
  is sufficient or whether a broader invariant is needed.
- Is `TRITON_DUMP_MIR` / `LLVM_ENABLE_TIMING` / `LLVM_PASS_PLUGIN_PATH` /
  `TRITON_ENABLE_ASAN` ever expected to be enabled in production? If these
  are strictly developer-only, issues #2 and #3 can be triaged as
  lower-priority dev-only races.
- Confirm by reading LLVM headers that `llvm::parallel::strategy` is indeed a
  plain non-atomic global, and not (for instance) an
  `std::atomic<llvm::ThreadPoolStrategy>`. If it is atomic, issue #4 drops
  to Minor.
