---
name: LLVM TimePasses global flags and reportAndResetTimings raced by concurrent compiles
description: translateLLVMIRToASM unconditionally writes the process-global llvm::TimePassesIsEnabled / TimePassesPerRun flags and calls reportAndResetTimings on the shared TimePassesHandler with the GIL released, so concurrent Triton compiles corrupt each other's timing state and produce interleaved or truncated reports on llvm::dbgs()
type: issue
---

# LLVM `TimePasses` global flags and `reportAndResetTimings` raced by concurrent compiles

- **Status:** Open
- **Severity:** Significant
- **Component:** `python/src/llvm.cc` — `translateLLVMIRToASM` (only active
  when `LLVM_ENABLE_TIMING` is set)
- **Tier:** 1/2

- **Shared state:**
  - `llvm::TimePassesIsEnabled` and `llvm::TimePassesPerRun` — process-global
    `bool`s that gate LLVM's pass-timing instrumentation.
  - LLVM's internal `TimePassesHandler` counters reset/flushed by
    `llvm::reportAndResetTimings`.
  - `llvm::dbgs()` — the process-global debug `raw_ostream` the timing
    report is appended to.
- **Writer(s):** `translateLLVMIRToASM` in `llvm.cc`. When
  `LLVM_ENABLE_TIMING` is on it does:
  - `llvm::TimePassesIsEnabled = true; llvm::TimePassesPerRun = true;`
    (plain writes, no synchronization)
  - calls `reportAndResetTimings(&reportStream)` twice (after the IR
    pipeline and after the codegen pipeline), then writes
    `llvm::dbgs() << reportStream.str();`.

  This is reached from the `translate_to_asm` pybind11 binding, which opens
  `py::gil_scoped_release allow_threads;` before calling
  `translateLLVMIRToASM` — so the body runs with the Python thread state
  detached. Backends invoke it once per kernel via `make_ptx` /
  `make_amdgcn` (`third_party/nvidia/backend/compiler.py:458`,
  `third_party/amd/backend/compiler.py:532`).
- **Reader(s):** every concurrent `translate_to_asm` call (both flags and
  the timer counters), plus any LLVM machinery that consults
  `TimePassesIsEnabled` to decide whether to record per-pass time.

- **Race scenario:** Thread A enters `translate_to_asm`, sets the global
  flags to `true`, runs its IR pipeline, and calls `reportAndResetTimings`
  to flush its own pass timings. Thread B enters `translate_to_asm`
  concurrently. Three things can interleave:
  1. Both threads write the same `true` value to the flags concurrently —
     a same-value race, technically UB but converges.
  2. B's `reportAndResetTimings` in the middle of A's IR pipeline drains
     A's in-progress counters, so A's eventual report is missing the
     time it had already accumulated.
  3. Both threads append to `llvm::dbgs()` (a single shared
     `raw_ostream`) with no synchronization, producing interleaved or
     torn timing-report output.

- **Suggested fix:** This is a diagnostic-only path, so two reasonable
  options:
  1. Serialize the timing-enabled region under the same process-wide
     compile mutex used for the `cl::opt` mutations
     (`issues/llvm/global-cl-opt-mutation.md`). Both concerns share the
     same root cause: process-global LLVM state mutated from a
     GIL-released hot path.
  2. Set `TimePassesIsEnabled` / `TimePassesPerRun` once at module-init
     time based on the env var instead of every compile, and skip the
     `reportAndResetTimings` calls when the flags aren't on.

  Either is acceptable; option 1 is the minimal change.
