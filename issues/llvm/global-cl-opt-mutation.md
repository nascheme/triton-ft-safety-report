---
name: Global LLVM cl::opt mutation from GIL-released compile paths
description: Process-wide llvm::cl::opt<> globals are mutated and "restored" from concurrent GIL-released compile bindings; the RAII save/restore pairing is broken under interleaving and the mutations themselves are unsynchronized
type: issue
---

# Global LLVM `cl::opt<>` mutation from GIL-released compile paths

- **Status:** Open
- **Severity:** SEVERE
- **Component:** `python/src/llvm.cc` — `setLLVMOption` / `restoreLLVMOption` / `ScopedLLVMOption`

- **Shared state:** the process-global `llvm::cl::opt<T>` registry returned by
  `llvm::cl::getRegisteredOptions()` and the individual `cl::opt<T>` objects it
  holds (e.g. `print-after-all`, `stop-before`, `stop-after`, `start-before`,
  `enable-misched`, `enable-post-misched`, `misched-print-dags`, plus any flag
  forwarded through `DISABLE_LLVM_OPT` / `flags`).
- **Writer(s):** `setLLVMOption<bool|string>` (llvm.cc:59, 77),
  `restoreLLVMOption<bool|string>` (llvm.cc:92, 102), and
  `ScopedLLVMOption<T>` (llvm.cc:112). They are called from
  `translateLLVMIRToASM` (llvm.cc:336, 340), `translateLLVMIRToMIR`
  (llvm.cc:255, 266, 272, 276), `translateMIRToASM` (llvm.cc:420–432),
  `dumpSchedulingDAG` (llvm.cc:163, 174, 182–184), and the `optimize_module`
  binding lambda (llvm.cc:656, 678). All five run with the Python thread state
  detached: `optimize_module` and `to_module` use
  `py::call_guard<py::gil_scoped_release>()` (llvm.cc:622, 759); the others
  open an explicit `py::gil_scoped_release allow_threads` (llvm.cc:769, 797,
  821, 849).
- **Reader(s):** the LLVM legacy and new pass managers themselves, which read
  these options by name during pipeline construction and execution (e.g.
  `MachineScheduler` checks `EnableMachineSched.getNumOccurrences()`,
  printers check `PrintAfterAll`).

- **Race scenario A (mutation race):** Thread A enters `translate_to_asm` and
  calls `setLLVMOption<bool>("print-after-all", true)`. Thread B concurrently
  enters `translate_mir_to_asm` and constructs
  `ScopedLLVMOption<std::string>("start-before", "machine-scheduler")`. Both
  threads mutate the same global registry/`cl::opt` objects with no
  synchronization — `addOccurrence` increments an unguarded counter and writes
  the value field; `setValue` overwrites without ordering. At minimum, B's
  flags land inside A's pipeline run, steering codegen down a different path
  (different stop/start pass, different scheduler choice) and producing wrong
  generated code. At worst, the unsynchronized writes corrupt the `cl::opt`
  internals.

- **Race scenario B (RAII save/restore inversion):** Thread A constructs
  `ScopedLLVMOption<bool> g("misched-print-dags", true)`, capturing
  `originalValue = false`. Before A's scope exits, Thread B constructs the
  same scope, observes `true` (set by A), and saves `originalValue = true`.
  When B's scope exits it "restores" to `true`, then A's scope exits and
  restores to `false`. The intermediate window leaks A's value to B's
  pipeline; in symmetric interleavings the option can also stay permanently
  flipped after both threads return. The save/restore invariant is logically
  wrong under any concurrent caller, independent of the C++ memory model.

- **Suggested fix:** Add a single process-wide mutex that covers the entire
  capture-set-run-restore body of every function that touches `cl::opt`
  state (`translateLLVMIRToASM`, `translateLLVMIRToMIR`, `translateMIRToASM`,
  `dumpSchedulingDAG`, and the `optimize_module` lambda). The lock must span
  `ScopedLLVMOption` construction through destruction so the original-value
  capture is accurate. Gate it under `#ifdef Py_GIL_DISABLED` to avoid cost
  in GIL-enabled builds. The GIL is already released, so there is no
  GIL/lock-ordering deadlock risk. Per-option locking does not fix scenario
  B — the critical section must be the whole scope.

  Long-term: migrate flag forwarding off `cl::opt` (e.g. new pass-manager
  `PipelineTuningOptions` and instrumentation callbacks) so per-compile
  configuration is no longer process-global. See `issues.md` triage notes
  for the extended discussion.
