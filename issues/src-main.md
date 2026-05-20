# python/src/main.cc Free-Threading Issues

## Issues

| # | Severity | Tier | Component | Issue |
|---|----------|------|-----------|-------|
| 1 | Minor | 1 | `init_triton_stacktrace_hook` → `llvm::sys::AddSignalHandler` | Process-global LLVM signal handler registered at import; env var `TRITON_ENABLE_PYTHON_STACKTRACE` is sampled once and the handler list is process-wide mutable state that other LLVM code may also write — worth a spot-check during the deeper pass but not expected to be a race under normal use |
| 2 | Not worth reporting | — | `PYBIND11_MODULE(libtriton, m)` body | All submodule wiring (`init_triton_ir`, `init_triton_llvm`, `init_native_specialize`, `init_triton_interpreter`, `init_triton_passes`, `init_gluon_ir`, `init_gsan_testing`, `init_linear_layout`, per-backend `init_triton_<name>`) runs under Python's import lock — write-once module init, covered by CLAUDE.md "low-value patterns" |

Issues in `python/src/main.cc` — the top-level pybind11 module entry point
for `triton._C.libtriton`. This file is pass-3 item #4 in the audit plan
(below `specialize.cc`, `llvm.cc`, `ir.cc`) because it contains almost no
logic of its own: it only declares submodule initializers and invokes them
from a single `PYBIND11_MODULE` body.

No separate issue file is written for item #1 at this stage — it is kept in
the table as a traceability note for the deeper C++ audit to confirm or
promote. Item #2 is kept in the table only to document that the reviewer
considered module-init wiring and explicitly ruled it out. See the triage
notes for reasoning and for items that were considered and rejected.

## Context

### What `main.cc` actually contains

The file is ~65 lines and consists of:

- Preprocessor plumbing (`FOR_EACH_*`, `FOR_EACH_P`, `DECLARE_BACKEND`,
  `INIT_BACKEND`) used to expand the CMake-defined `TRITON_BACKENDS_TUPLE`
  macro (see `triton/CMakeLists.txt` line 413-414) into one
  `init_triton_<name>(...)` declaration and one call per enabled backend.
- Forward declarations for the submodule initializers defined in the
  sibling `.cc` files and in the `gsan_testing` / backend subtrees.
- A single `PYBIND11_MODULE(libtriton, m)` body (lines 51–64) that:
  1. Sets the module docstring.
  2. Calls `init_triton_stacktrace_hook(m)` — registers an LLVM signal
     handler if `TRITON_ENABLE_PYTHON_STACKTRACE` is set.
  3. Calls `init_triton_env_vars(m)` — registers env-var helper
     bindings (implementation in `ir.cc` line 2061).
  4. Calls `init_native_specialize(m)` — registers the
     `native_specialize_impl` / `make_tensordesc_args` module methods via
     `PyModule_AddFunctions` (implementation in `specialize.cc` line 764).
  5. Creates the `ir`, `passes`, `interpreter`, `llvm`, `gsan_testing`,
     `linear_layout`, `gluon_ir` submodules via `m.def_submodule(...)` and
     calls the matching `init_*` for each.
  6. Expands `FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)` into one
     `init_triton_<name>(m.def_submodule(#name))` call per backend (for
     example `init_triton_nvidia(m.def_submodule("nvidia"))`).

### What shared mutable state is introduced *by this file*

None of its own. `main.cc` does not define any file-scope mutable state, no
static caches, no lazy initializers, no hooks, and no `py::gil_scoped_release`
scopes. Everything it does is either:

- preprocessor expansion (compile-time only), or
- calling sibling `init_*` functions from the one-shot `PYBIND11_MODULE`
  body, which by construction runs under CPython's import lock.

Under free-threading, `PYBIND11_MODULE` initialization still holds the
import lock, so the call sequence in lines 52–63 executes single-threaded
with respect to other imports of the same module. This matches the
"write-once globals set during module init" pattern that `CLAUDE.md`
explicitly calls out as low-value.

The shared mutable state actually wired up here lives in the callees, and
each callee belongs to a separate audit report:

| Callee                             | Defined in                  | Audit report              |
|------------------------------------|-----------------------------|---------------------------|
| `init_triton_stacktrace_hook`      | `python/src/llvm.cc:924`    | (note in this file, see triage #1) |
| `init_triton_env_vars`             | `python/src/ir.cc:2061`     | `ir/issues.md`            |
| `init_native_specialize`           | `python/src/specialize.cc:764` | `specialize/issues.md` |
| `init_triton_ir`                   | `python/src/ir.cc`          | `ir/issues.md`            |
| `init_triton_passes`               | `python/src/passes.cc`      | pass 3, separate report   |
| `init_triton_interpreter`          | `python/src/interpreter.cc` | pass 4                    |
| `init_triton_llvm`                 | `python/src/llvm.cc`        | `llvm/issues.md`          |
| `init_gsan_testing`                | `python/triton/experimental/gsan/src/gsan_testing.cc` | pass 5 |
| `init_linear_layout`               | `python/src/linear_layout.cc` | pass 4                  |
| `init_gluon_ir`                    | `python/src/gluon_ir.cc`    | pass 4                    |
| `init_triton_<backend>`            | per-backend subtree         | e.g. `nvidia-driver/`     |

## Triage notes

### 1. `init_triton_stacktrace_hook` — `llvm::sys::AddSignalHandler` (Minor)

- **Shared state:** (a) the process-wide LLVM signal-handler list that
  `llvm::sys::AddSignalHandler` appends to, and (b) the ambient
  process-signal disposition that the handler alters when it fires. The
  handler function (`llvm.cc` lines 919–922) prints the LLVM stack trace
  and then `raise(SIGABRT)`.
- **Writer:** `init_triton_stacktrace_hook(m)` called exactly once from the
  `PYBIND11_MODULE` body, under the import lock (`main.cc` line 53).
- **Reader / invoker:** LLVM's signal-handling machinery, fired from
  process signal context — not from free-threaded Python paths.
- **Why this is only Minor:** the registration itself is write-once under
  the import lock, so registration is not racing anything. The handler list
  mutation is a concern only if some *other* code path later calls
  `AddSignalHandler` / `RemoveSignalHandler` / `RunSignalHandlers` from a
  Triton-controlled thread while Python is multi-threaded. That would be a
  question for the `llvm.cc` audit, not for `main.cc`.
- **Note for the deeper pass:** `TRITON_ENABLE_PYTHON_STACKTRACE` is
  sampled exactly once, at module import. Mutating the env var later has no
  effect. That is a design choice, not a race, and is consistent with how
  other Triton env-var knobs behave — worth noting for completeness, not
  worth reporting as a bug.
- **Fix direction:** none needed from `main.cc`. If the deeper `llvm.cc`
  pass finds a concurrent-mutation path on the LLVM handler list, fix it
  there.

### 2. `PYBIND11_MODULE(libtriton, m)` body (Not worth reporting)

- **Shared state:** the `libtriton` module object, its submodules, and any
  state registered by the called `init_*` functions.
- **Concern:** whether two threads can race on the module-init sequence
  itself.
- **Why this is not worth reporting:** `PYBIND11_MODULE` is Python's module
  init entry point, invoked by CPython's import machinery while holding the
  import lock. Two interpreter threads cannot execute the body
  concurrently. This is exactly the "write-once globals set during module
  init" pattern that `CLAUDE.md` says not to report.
- **Only caveat:** the *callees* can install shared mutable state that is
  *later* mutated on the hot path (the specialize.cc lazy-init TOCTOU is the
  canonical example). Those issues are findings against the callees, not
  against `main.cc`. They are tracked in the per-callee audit reports
  listed in the Context section table.

## Rejected / low-value observations

- **Macro expansion (`FOR_EACH_*`, `FOR_EACH_P`, `DECLARE_BACKEND`,
  `INIT_BACKEND`).** Preprocessor only. No runtime effect and no shared
  state.

- **Order of `init_*` calls inside `PYBIND11_MODULE`.** Deterministic,
  single-threaded under the import lock. Reordering is a correctness concern
  (e.g., `init_triton_stacktrace_hook` before other init wants the handler
  in place first), not a concurrency concern.

- **`m.def_submodule(...)` return-value lifetime.** Submodule objects are
  owned by the parent module and live for the life of the interpreter.
  The `init_triton_*(m.def_submodule(...))` calls receive `pybind11::module
  &&` / `pybind11::module &` and do not retain the rvalue beyond the init
  call. Not a lifetime issue.

- **Forward declarations of `init_triton_*` functions.** Pure C++
  declarations; resolved at link time. No runtime effect.

- **Backend init via `TRITON_BACKENDS_TUPLE`.** Each enabled backend's
  `init_triton_<name>` is called exactly once here, under the import lock.
  Backend-internal shared mutable state (for example, NVIDIA driver
  globals) is the concern of the corresponding backend audit
  (`nvidia-driver/`, etc.), not of `main.cc`.

- **`m.doc() = ...`.** Write-once module attribute at init time.

## Open questions for the deeper audit

- Confirm that no other caller of `llvm::sys::AddSignalHandler` runs after
  `init_triton_stacktrace_hook` from a non-main thread. If all
  `AddSignalHandler` calls happen at import or before spawning worker
  threads, item #1 is fully discharged.
- Confirm that `TRITON_ENABLE_PYTHON_STACKTRACE` is documented as
  import-time-only; if users expect it to be re-readable, that is a docs
  issue, not a thread-safety issue.
- Cross-check the per-backend `init_triton_<name>` entries identified via
  `TRITON_BACKENDS_TUPLE` expansion against the backend subtree audits to
  make sure none of them silently hold shared mutable state that was missed
  because this file only shows a macro-expanded call site.
