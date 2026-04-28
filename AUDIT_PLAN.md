# Triton Free-Threading Audit Plan: Scope and Ordering

See [CLAUDE.md](CLAUDE.md) for audit methodology, high-value race patterns,
report format, and severity guidance.

See [OVERVIEW.md](OVERVIEW.md) for the high-level architecture of Triton's
Python-facing runtime.

## Scope

Triton source is symlinked at `./triton`.

The audit target is the **Python boundary**: code directly involved in
Python-level kernel definition, specialization, compilation orchestration,
launch, caching, backend/driver selection, and Python/C++ interaction.

Primary scope:

- `triton/python/triton/`
  - runtime dispatch and launch (`runtime/`)
  - compiler orchestration (`compiler/`)
  - backend discovery/interfaces (`backends/`)
  - process-global config and hooks (`knobs.py`)
  - selected language/tooling surfaces when they hold shared mutable state or
    participate in the hot compile/launch path
- `triton/python/src/`
  - pybind11 extension sources for `triton._C.libtriton`

Out of scope unless they directly affect Python-visible state or execute in
GIL-released regions:

- deep MLIR/LLVM optimization internals
- backend codegen correctness unrelated to shared mutable state
- pure compiler-pass logic that does not touch Python objects, global runtime
  state, caches, lazy initialization, or runtime-visible native state

## Audit goal

The goal is **not** to re-verify all compiler correctness.

The goal is to find places where free-threaded Python can expose bugs in:

- process-global mutable state
- mutable state attached to shared wrapper objects
- lazy initialization / first-use setup
- cache population and cache lookup paths
- async/future bookkeeping and finalization
- Python/C API usage in native helpers
- pybind11 bindings that run without the GIL

When in doubt, prefer:
- identifying the shared state,
- tracing writers/readers,
- and leaving a precise follow-up note

rather than guessing about behavior that has not yet been verified.

## Recommended audit order

Start with the hot dispatch/compile path, then move outward to supporting state
and lower-frequency surfaces.

---

## Pass 1: Core Python runtime dispatch path

These files define the most important shared wrapper objects and the common path
for kernel invocation.

1. `triton/python/triton/runtime/jit.py`
   - `JITFunction`
   - specialization/binder creation
   - per-device caches
   - global JIT-function registry
   - preload / async compile integration

2. `triton/python/triton/compiler/compiler.py`
   - `compile(...)`
   - backend selection
   - cache-manager interaction
   - `CompiledKernel`
   - lazy runtime handle initialization

3. `triton/python/triton/runtime/driver.py`
   - active/default driver singleton
   - lazy driver selection

4. `triton/python/triton/runtime/autotuner.py`
   - autotune wrapper state
   - tuning-result caches
   - benchmarking path layered on top of `JITFunction`

Why first:
- these files define the hottest Python execution path
- they contain the most important shared mutable Python objects
- they directly cover tier-1 and tier-2 concurrency scenarios

---

## Pass 2: Global support state used by the runtime

These files may be reached by many launches/compiles even if they are not the
center of dispatch themselves.

1. `triton/python/triton/knobs.py`
   - process-global mutable configuration objects
   - hook chains / callback slots
   - mutation during compile/load/launch

2. `triton/python/triton/runtime/cache.py`
   - compiler artifact cache managers
   - cache-key helpers
   - local/remote cache coordination

3. `triton/python/triton/runtime/_async_compile.py`
   - `AsyncCompileMode`
   - `FutureKernel`
   - async bookkeeping/finalization

4. `triton/python/triton/backends/__init__.py`
   - backend discovery
   - backend registry creation

5. `triton/python/triton/backends/compiler.py`
6. `triton/python/triton/backends/driver.py`
   - base interfaces used to reason about backend/driver lifecycle and what
     runtime methods may touch shared state

Why second:
- these files explain the global state surrounding dispatch
- they are important for tier-1 and tier-3 issues
- they often determine whether a race is local to one object or process-global

---

## Pass 3: Python/C++ boundary on the hot path

This is the highest-priority native pass. Focus on bindings and helpers that are
reachable from normal specialization/compile flow.

1. `triton/python/src/specialize.cc`
   - highest-priority C++ file
   - runtime argument specialization helper used from `runtime/jit.py`
   - lazy static initialization and native caches directly matter here

2. `triton/python/src/llvm.cc`
   - GIL-released compilation regions
   - long-running native work reachable during compilation

3. `triton/python/src/ir.cc`
   - MLIR bindings
   - explicit GIL-release sites

4. `triton/python/src/main.cc`
   - module init / exported submodules / wiring

5. `triton/python/src/passes.cc`
   - pass bindings exposed to Python

Why third:
- once the Python runtime flow is understood, these files are the next highest
  value place to look for real races
- this pass is where lazy native state, `PyObject*` lifetime assumptions, and
  GIL-release issues become concrete

---

## Pass 4: Lower-frequency runtime/compiler support surfaces

These files are relevant, but usually only after the main dispatch/cache story
is clear.

1. `triton/python/triton/runtime/build.py`
   - builds cached native helper modules from generated C code

2. `triton/python/triton/compiler/make_launcher.py`
   - launcher-code generation used with runtime build helpers

3. `triton/python/triton/compiler/code_generator.py`
   - front-end lowering path; audit only for Python-visible shared state,
     module-global caches, or mutable module-level objects

4. `triton/python/triton/runtime/interpreter.py`
   - interpreter-mode runtime path

5. `triton/python/src/interpreter.cc`
   - native interpreter helpers and existing synchronization choices

6. `triton/python/triton/runtime/_allocation.py`
   - context-local allocator state; mostly lower priority, but verify it stays
     context-local and does not escape into shared mutable globals

7. `triton/python/src/linear_layout.cc`
8. `triton/python/src/gluon_ir.cc`
   - lower-frequency native bindings

Why fourth:
- these matter for completeness
- they can contain real issues, but they are less central than the core
  specialization / compile / launch path

---

## Pass 5: Peripheral tools, language helpers, and experimental surfaces

Audit these after the runtime/compiler boundary is understood.

1. `triton/python/triton/language/core.py`
   - only for shared mutable state, caching decorators, or module-global objects
   - do not turn this into a full DSL semantics audit

2. `triton/python/triton/tools/compile.py`
3. `triton/python/triton/tools/link.py`
4. `triton/python/triton/tools/disasm.py`
   - usually lower risk; mainly check for global caches or shared mutable state

5. `triton/python/triton/experimental/gluon/`
   - audit if later evidence suggests it shares the same runtime/cache/global
     patterns as the main Triton path

Why fifth:
- these surfaces are real, but they are less central than the main runtime path
- they are best audited after the core concurrency model is already clear

## Concrete questions to carry through every pass

As you move through the passes, keep these questions in mind:

1. **What state is shared?**
   - module globals
   - singleton objects
   - mutable attributes on reusable wrapper objects
   - dict/defaultdict/list/native-container caches

2. **Who writes it, and when?**
   - import time
   - first use
   - cache fill
   - callback/finalization path
   - explicit configuration mutation

3. **Who reads it, and on what path?**
   - dispatch hot path
   - compile path
   - lazy load path
   - autotune path
   - callback or hook execution

4. **Can those accesses happen concurrently?**
   - same `JITFunction`
   - same `Autotuner`
   - same `CompiledKernel`
   - one thread compiling while another launches
   - one thread mutating knobs/hooks while another executes

5. **Does the code cross the Python/C++ boundary while assuming GIL-style serialization?**
   - `native_specialize_impl`
   - pybind11 `call_guard<py::gil_scoped_release>()`
   - explicit `py::gil_scoped_release` scopes
   - native containers storing Python object pointers

6. **What is the concrete consequence?**
   - corrupted Python/C++ container
   - duplicate compile/finalization
   - stale or wrong cached result
   - wrong backend/driver/config selection
   - crash, use-after-free, or wrong-kernel behavior
   - or only benign duplicate work / stale read

## Tier mapping to keep in mind

Use the concurrency tiers from [CLAUDE.md](CLAUDE.md) while auditing:

- **Tier 1:** one thread uses Triton while other threads do unrelated Python work
- **Tier 2:** multiple threads call the same Triton kernel/autotuner concurrently
- **Tier 3:** one thread mutates Triton configuration while another executes

A good report should usually say which tier an issue belongs to.

## Suggested output structure for later reports

When you turn a pass into a report, keep it narrowly scoped by subsystem, for
example:

- `jit/issues.md`
- `autotuner/issues.md`
- `driver-and-knobs/issues.md`
- `compiler/issues.md`
- `specialize/issues.md`
- `llvm/issues.md`

That keeps architecture summaries short and makes it easier for later review to
confirm specific claims file by file.
