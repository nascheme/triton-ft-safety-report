---
name: Global LLVM cl::opt mutation from GIL-released compile paths
description: Process-wide llvm::cl::opt<> globals are mutated and "restored" from concurrent GIL-released compile bindings; the RAII save/restore pairing is broken under interleaving and the mutations themselves are unsynchronized
type: issue
---

# Global LLVM `cl::opt<>` mutation from GIL-released compile paths

- **Issue-Id:** FT029
- **Status:** Open
- **Severity:** LOW (pre-existing maintainer-rule violation, not a free-threading bug)
- **Component:** `python/src/llvm.cc` — `setLLVMOption` / `restoreLLVMOption` / `ScopedLLVMOption`
- **Tier:** Out-of-scope (see README "Pre-existing rule violations" /
  Tier 3 family)

> **Scope note:** Triton maintainers' rule is "don't mutate `cl::opt`
> per-kernel"; only debug features currently do it. The race exists under
> the GIL build too. Per `README.md` and `CLAUDE.md`, this category is
> tracked for completeness rather than promoted to a free-threading tier.
> Filed here so the failure mode is documented, not as an ask to fix.

- **Shared state:** the process-global `llvm::cl::opt<T>` registry returned by
  `llvm::cl::getRegisteredOptions()` and the individual `cl::opt<T>` objects it
  holds (e.g. `print-after-all`, `stop-before`, `stop-after`, `start-before`,
  `enable-misched`, `enable-post-misched`, `misched-print-dags`, plus any flag
  forwarded through `DISABLE_LLVM_OPT` / `flags`).
- **Writer(s):** `setLLVMOption<bool|string>`, `restoreLLVMOption<bool|string>`,
  and `ScopedLLVMOption<T>` in `llvm.cc`. They are called from
  `translateLLVMIRToASM`, `translateLLVMIRToMIR`, `translateMIRToASM`,
  `dumpSchedulingDAG`, and the `optimize_module` binding lambda. All five
  run with the Python thread state detached: `optimize_module` and
  `to_module` use `py::call_guard<py::gil_scoped_release>()`; the others
  open an explicit `py::gil_scoped_release allow_threads`.
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

- **Suggested fix:** Honor the existing maintainer rule and stop mutating
  `cl::opt` from runtime paths. Where the only callers are debug features
  (`TRITON_DUMP_MIR`, `LLVM_IR_ENABLE_DUMP`, flag forwarding), document
  that those features are single-threaded. If concurrent in-process use of
  any of these debug paths becomes a goal, the minimal fix is a single
  process-wide mutex that covers the entire capture-set-run-restore body
  of every function that touches `cl::opt` state (gate under
  `#ifdef Py_GIL_DISABLED`); per-option locking does not fix scenario B.
  See `issues/llvm.md` triage notes for the extended discussion.
