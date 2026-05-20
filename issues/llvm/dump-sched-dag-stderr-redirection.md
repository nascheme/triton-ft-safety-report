---
name: dumpSchedulingDAG stderr redirection races process-wide stderr
description: dumpSchedulingDAG saves, replaces, and restores the process-wide stderr FD with no synchronization while running with the GIL released; concurrent compiles or other Python stderr writers see the redirected FD or have it restored to the wrong descriptor
type: issue
---

# `dumpSchedulingDAG` stderr redirection races process-wide stderr

- **Status:** Open
- **Severity:** SEVERE
- **Component:** `python/src/llvm.cc` — `dumpSchedulingDAG` (only active when
  `TRITON_DUMP_MIR` is set)

- **Shared state:** the process's `stderr` `FILE*` and file-descriptor 2.
  Both are process-global and shared with every other thread, including all
  Python code that writes to `sys.stderr`.
- **Writer(s):** `dumpSchedulingDAG` (llvm.cc:203–231):
  - `int saved_stderr_fd = dup(fileno(stderr));` (line 204)
  - `FILE *redirected = freopen(dumpFilename.c_str(), "a", stderr);` (line 207)
  - run the codegen pipeline (`pass.run(module)`, line 222) which emits the
    scheduling DAG via LLVM debug printing onto `stderr`
  - `fflush(stderr); dup2(saved_stderr_fd, fileno(stderr)); close(saved_stderr_fd);`
    (lines 226–229)

  Reached from the `dump_sched_dag` pybind11 binding (llvm.cc:792), which
  opens `py::gil_scoped_release allow_threads;` at line 797 before calling
  `dumpSchedulingDAG` — so the body runs with the Python thread state
  detached.
- **Reader(s):** every thread in the process that reads or writes `stderr` /
  fd 2 during this window: other concurrent `dumpSchedulingDAG` calls, LLVM
  diagnostics from any other compile path (`llvm::errs()`), and any Python
  code touching `sys.stderr`.

- **Race scenario A (interleaved redirect/restore):** Thread A executes line
  204, capturing `saved_stderr_fd = X` (the real terminal/log fd). Thread A
  executes line 207, pointing `stderr` at `fileA.txt` (now `fileno(stderr)`
  is some new fd `Y`). Before A reaches line 228, Thread B enters
  `dumpSchedulingDAG`, calls `dup(fileno(stderr))` and captures
  `saved_stderr_fd = Y'` — i.e. a dup of A's redirected stderr, not the
  original. B then `freopen`s onto `fileB.txt`. A reaches line 228 and
  `dup2`s `X` back onto fd 2, restoring the real stderr while B is still
  mid-pipeline; B's subsequent DAG output goes to the real stderr instead of
  `fileB.txt`. When B finally restores, it `dup2`s `Y'` (a duplicate of A's
  intermediate redirected FILE) onto fd 2, leaving the process pointed at a
  stale dump-file fd permanently after both threads return.

- **Race scenario B (collateral damage to unrelated stderr writers):** While
  Thread A holds the redirect open, any other thread — a different compile
  path calling `llvm::errs()`, Python logging, `print(..., file=sys.stderr)`
  — writes into `fileA.txt` instead of stderr. This is not just lost
  output; arbitrary process diagnostics get appended to the MIR dump file.

- **Suggested fix:** Serialize the entire redirect-run-restore region under
  a dedicated mutex shared with any other code that touches process-wide
  stderr (in practice, just this function). Gate under
  `#ifdef Py_GIL_DISABLED` if desired — but note that the race also exists
  in GIL-enabled builds when threaded compile workers are used, since the
  GIL is released across the body. The mutex still does not stop unrelated
  threads from being silently captured (scenario B).

  Better fix: stop redirecting process-wide `stderr` entirely. Capture the
  scheduling-DAG output through an LLVM `raw_fd_ostream` / `raw_string_ostream`
  attached to the legacy pass manager, or set `DebugFlag` / scoped
  `llvm::dbgs()` redirection if available, so the dump is written without
  touching fd 2. This eliminates both scenarios.
