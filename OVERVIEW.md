# Triton Python Boundary Overview (for Free-Threading Audit)

Note: Triton source is available under `./triton`.

This document is intentionally **high-level**. It is meant to give later audit passes
an accurate map of how Triton's Python-facing runtime is structured, without trying
to prove every race or trace every call edge in detail.

Use it as an architecture map for later audits: it should help you identify the
hot path, the major shared wrapper objects, the major process-global state, and
the main Python/C++ boundaries worth deeper inspection.

Triton `3.6.0` is a Python DSL and JIT compiler for GPU kernels. At the Python
boundary, the important pieces are:

- the Python DSL and decorators (`triton.language`, `@triton.jit`, `@triton.autotune`)
- the Python runtime that specializes, compiles, caches, and launches kernels
- the pybind11 extension module `triton._C.libtriton`, which exposes MLIR/LLVM and
  specialization helpers to the Python layer
- backend/compiler/driver plugin interfaces used to target specific GPUs

## High-Level Execution Flow

A typical Triton kernel launch looks like this:

1. User writes a Python function with Triton DSL ops from `triton.language`.
2. `@triton.jit` wraps that function in a `JITFunction`.
3. Calling `fn[grid](...)` goes through `JITFunction.run(...)` in `runtime/jit.py`.
4. Triton:
   - queries the active runtime driver/device,
   - specializes the kernel signature for the actual runtime arguments,
   - builds a cache key from the function, argument specialization, options, and target,
   - looks in an in-memory per-device cache for an already-compiled kernel.
5. On a cache miss, Triton calls `triton.compiler.compile(...)`.
6. The compiler:
   - selects a backend implementation for the current GPU target,
   - lowers Python AST or input IR through a sequence of backend-defined stages,
   - stores intermediate/final artifacts plus metadata in the cache,
   - returns a `CompiledKernel` handle.
7. On first real use, `CompiledKernel` lazily loads the compiled binary through the
   active driver and creates the launcher/runtime handles needed to execute it.
8. Later launches usually reuse cached specialization results and cached compiled
   kernels.

`@triton.autotune` sits above this path: it wraps a kernel-like object, benchmarks
multiple configurations, remembers the best one for a given argument signature,
and then dispatches through the underlying `JITFunction`.

## Architecture Sketch

```text
User Python kernel
  -> triton.language DSL
  -> @triton.jit / JITFunction                    (python/triton/runtime/jit.py)
     -> runtime specialization + per-device cache
     -> triton.compiler.compile(...)             (python/triton/compiler/compiler.py)
        -> backend-selected lowering stages
        -> triton._C.libtriton                   (python/src/*.cc via pybind11)
           -> MLIR / LLVM / specialization helpers
        -> cached artifacts + metadata
     -> CompiledKernel
        -> lazy load via active driver
        -> launcher executes GPU binary
```

## Python-Relevant Directory Map

```text
python/
  src/                          # C++ pybind11 sources for triton._C.libtriton
    main.cc                     #   module entry point
    ir.cc                       #   MLIR IR bindings
    llvm.cc                     #   LLVM bindings / codegen helpers
    passes.cc                   #   MLIR pass bindings
    specialize.cc               #   Python-argument specialization helper
    interpreter.cc              #   interpreter-mode helpers / atomics
    gluon_ir.cc                 #   experimental Gluon IR bindings
    linear_layout.cc            #   linear layout bindings

  triton/
    __init__.py                 # Public Python API surface
    knobs.py                    # Process-global configuration objects
    errors.py                   # Common exception types
    _utils.py                   # Shared Python helpers

    runtime/
      jit.py                    # JITFunction, specialization, dispatch, in-memory kernel caches
      autotuner.py              # Autotuner wrapper and config selection cache
      driver.py                 # Active/default driver singleton used by the Python runtime
      cache.py                  # On-disk / remote cache managers and cache-key helpers
      build.py                  # Builds small cached native helper modules from generated C code
      _async_compile.py         # AsyncCompileMode / FutureKernel
      _allocation.py            # ContextVar-based allocator state
      interpreter.py            # Python-side interpreter runtime support

    compiler/
      compiler.py               # Main compile entry point and CompiledKernel
      code_generator.py         # Python AST -> Triton IR lowering
      make_launcher.py          # Generates launcher C code used by runtime helpers
      errors.py                 # Compiler-specific exceptions

    language/
      core.py                   # Core Triton DSL types/ops
      semantic.py               # Semantic lowering helpers
      standard.py               # Standard language helpers
      math.py, random.py        # Additional DSL functions
      extra/                    # Extra language functionality

    backends/
      __init__.py               # Backend discovery/registry
      compiler.py               # Base backend compiler interface
      driver.py                 # Base backend driver interface

    experimental/gluon/         # Experimental Gluon frontend/runtime pieces
    tools/                      # Disassembly and utility helpers
```

## Important Runtime Components

| Component | Location | Role |
|---|---|---|
| `JITFunction` | `runtime/jit.py` | Wraps a `@triton.jit` function; handles specialization, caching, compilation, and launch |
| `KernelParam` | `runtime/jit.py` | Stores per-parameter metadata derived from the Python signature |
| `Autotuner` | `runtime/autotuner.py` | Benchmarks multiple configs and caches the best choice |
| `DriverConfig` | `runtime/driver.py` | Holds the runtime's default/active backend driver instance |
| `CompiledKernel` | `compiler/compiler.py` | Represents a compiled kernel plus cached artifacts and lazy runtime handles |
| Cache managers | `runtime/cache.py` | Manage local/remote cache storage for compiler artifacts |
| `AsyncCompileMode` / `FutureKernel` | `runtime/_async_compile.py` | Coordinate background compilation and later finalization |
| Backend registry | `backends/__init__.py` | Discovers backend/compiler + driver plugins |
| `triton._C.libtriton` | `python/src/*.cc` | Pybind11 extension exposing MLIR/LLVM/specialization functionality |

## Backend Model

Triton's Python runtime does not hardcode one compiler/driver pair directly in
`runtime/jit.py` or `compiler/compiler.py`.

Instead:

- `triton.backends.compiler.BaseBackend` defines the compiler-side backend API.
- `triton.backends.driver.DriverBase` defines the runtime driver API.
- `triton.backends.__init__` discovers backend plugins and builds a registry.
- `runtime/driver.py` chooses the active driver.
- `compiler/compiler.py` chooses the matching compiler backend for the current
  `GPUTarget`.

So the runtime path is roughly: **active driver picks a target -> compiler picks a
matching backend -> backend supplies lowering stages and runtime metadata handling**.

## C++ Extension Module: `triton._C.libtriton`

`triton._C.libtriton` is the main Python/C++ boundary.

From `python/src/main.cc`, it exposes submodules/functions including:

- `ir` - MLIR IR creation/manipulation bindings
- `passes` - pass registration/bindings
- `llvm` - LLVM-related compilation helpers
- `interpreter` - interpreter helpers
- `gluon_ir` - experimental Gluon IR bindings
- `linear_layout` - layout bindings
- `native_specialize_impl` - fast C++ helper used by `runtime/jit.py` to classify
  and specialize Python arguments
- backend-specific submodules generated at build time

For free-threading, this module matters because it contains:

- Python C API usage,
- C++ static state,
- pybind11 bindings that may run with the GIL released.

## Where Caching Happens

There are several distinct caching layers:

1. **Per-`JITFunction`, per-device in-memory cache**
   - Stores compiled kernels/specializations used by dispatch.

2. **Autotuner in-memory cache**
   - Stores the best configuration for a tuning key.

3. **On-disk or remote compiler artifact cache**
   - Managed via `runtime/cache.py`.
   - Stores metadata and stage outputs for compiled kernels.

4. **Small helper-function caches**
   - `functools.lru_cache` is used in a few places such as compiler/cache/build
     helpers.

These layers are central to the free-threading audit because they are shared by
repeated launches and are often reached from hot paths. Later audits should
focus on who populates them, who reads them, and whether those accesses can
happen concurrently.

## Shared Mutable State Categories to Audit

At a high level, the Python-facing runtime has four main categories of shared
mutable state:

### 1. Process-global Python registries/singletons
Examples include:

- the global JIT-function registry in `runtime/jit.py`
- the runtime driver singleton in `runtime/driver.py`
- the backend registry in `backends/__init__.py`
- the global knob/config objects in `knobs.py`

### 2. Mutable state hanging off shared wrapper objects
Examples include:

- `JITFunction`'s per-device caches
- `Autotuner`'s tuning-result cache
- async compilation bookkeeping in `_async_compile.py`
- lazy runtime-handle initialization on `CompiledKernel`

These matter especially for tier-2 scenarios where multiple threads call the
same Triton kernel or autotuned wrapper concurrently.

### 3. Cache managers and filesystem-backed state
`runtime/cache.py` and `runtime/build.py` coordinate materialized artifacts,
cache directories, and helper-module build outputs. The filesystem gives some
atomicity guarantees for rename/replace, but the Python objects coordinating
those actions are still shared state.

### 4. C++ static state inside `libtriton`
Most importantly, `specialize.cc` contains lazy-initialized static Python object
pointers and C++ lookup tables used during runtime specialization. This area is a
high-priority audit target because it sits directly on the hot specialization /
JIT dispatch path.

## Context-Local vs Shared State

Not all mutable state is shared across threads.

Examples that are intentionally context-local in the current Python layer:

- `_allocation._allocator` in `runtime/_allocation.py` (`ContextVar`)
- `_async_compile.active_mode` in `runtime/_async_compile.py` (`ContextVar`)

These are lower priority than ordinary module globals or mutable attributes on
shared kernel wrapper objects.

## GIL-Released Regions Worth Remembering

The main pybind11 files with explicit GIL release points are:

- `python/src/ir.cc`
- `python/src/llvm.cc`

Those files deserve special attention because free-threaded Python removes the
usual assumption that Python-side activity is serialized while C++ work is
running.

## What This Overview Intentionally Does Not Do

This file is not trying to settle exact severity, prove every race, or document
every individual field. Later subsystem-specific audit notes should do that.

The purpose of this overview is just to keep the high-level map correct:

- where dispatch happens,
- where compilation happens,
- where backend selection happens,
- where the hot specialization / compile / launch path runs,
- where caching happens,
- and where shared mutable state broadly lives at the Python/C++ boundary.
