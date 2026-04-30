# Triton Free-Threading Safety Audit

Tracking thread-safety issues in Triton's Python layer for Python 3.14t (free-threading / nogil).

We are auditing the Python code under `python/` (both the pure-Python `triton/`
package and the C++ pybind11 extensions in `src/`). The C++/MLIR/LLVM core is
out of scope except where it interacts with the Python layer (GIL release
points, pybind11 bindings).

See [CLAUDE.md](CLAUDE.md) for the audit methodology.
See [OVERVIEW.md](OVERVIEW.md) for an architectural overview of the Triton codebase.


## Audit Progress

| Component | Status | SEVERE | Significant | Minor | Total |
|-----------|--------|--------|-------------|-------|-------|
| [jit](jit/issues.md) | In-review | 1 | 7 | 0 | 8 |
| [autotuner](autotuner/issues.md) | In-review | 1 | 4 | 2 | 7 |
| [compiler](compiler/issues.md) | In-review | 1 | 3 | 0 | 4 |
| [runtime-driver](runtime-driver/issues.md) | In-review | 1 | 3 | 0 | 4 |
| [nvidia-driver](nvidia-driver/issues.md) | In-review | 0 | 0 | 2 | 2 |
| [specialize](specialize/issues.md) | Ongoing | 4 | 1 | 1 | 6 |
| [async-compile](async-compile/issues.md) | Ongoing | 0 | 2 | 2 | 4 |
| [cache](cache/issues.md) | Ongoing | 0 | 0 | 4 | 4 |
| [knobs](knobs/issues.md) | Ongoing | 1 | 4 | 3 | 8 |
| [backends](backends/issues.md) | Ongoing | 0 | 0 | 2 | 2 |
| [ir](ir/issues.md) | Ongoing | 4 | 3 | 1 | 8 |
| [llvm](llvm/issues.md) | Ongoing | 2 | 5 | 1 | 8 |
| [src-main](src-main/issues.md) | Ongoing | 0 | 0 | 1 | 1 |
| [src-passes](src-passes/issues.md) | Ongoing | 0 | 2 | 2 | 4 |
| [runtime-build](runtime-build/issues.md) | Ongoing | 0 | 1 | 5 | 6 |
| [interpreter](interpreter/issues.md) | Ongoing | 2 | 2 | 3 | 7 |
| [compiler-codegen](compiler-codegen/issues.md) | Ongoing | 0 | 0 | 0 | 0 |
| [native-helpers](native-helpers/issues.md) | Ongoing | 1 | 3 | 1 | 5 |
| [language](language/issues.md) | Ongoing | 0 | 0 | 6 | 6 |
| [tools](tools/issues.md) | Ongoing | 0 | 0 | 4 | 4 |
| [experimental](experimental/issues.md) | Ongoing | 0 | 3 | 4 | 7 |
| **Total** | | **18** | **43** | **44** | **105** |

## Components

| Component | Description | Files |
|-----------|-------------|-------|
| jit | `@triton.jit` decorator, `JITFunction` class, kernel dispatch, specialization. Global `_triton_jit_function_registry` dict; per-instance `device_caches` defaultdict; `@cached_property` on `KernelParam`. | `runtime/jit.py` |
| autotuner | `@autotune` decorator, config benchmarking. Per-instance `cache` and `configs_timings` dicts; disk cache read/write. | `runtime/autotuner.py` |
| compiler | Main compilation pipeline orchestration. LRU-cached `max_shared_mem()`; `CompileTimer` accumulates timing data; `LazyDict`/`AsmDict` mutable compilation artifacts. | `compiler/compiler.py`, `compiler/code_generator.py`, `compiler/make_launcher.py` |
| runtime-driver | `DriverConfig` singleton managing active GPU driver. Lazy-init `_default`/`_active` properties without synchronization. | `runtime/driver.py` |
| nvidia-driver | NVIDIA backend driver implementation. | `backends/nvidia/driver.py` |
| specialize | Kernel argument type specialization. Static global pointers (`constexpr_cls`, etc.), static `dtype_ptr2str`/`dtype2str`/`type_handler_cache` maps; one-shot `init_globals()` without locking. | `python/src/specialize.cc` |
| async-compile | `AsyncCompileMode` context manager for background compilation via `concurrent.futures`. Uses `ContextVar` (safe), but `FutureKernel` result caching needs review. | `runtime/_async_compile.py` |
| cache | Cache manager abstractions (`CacheManager`, `FileCacheManager`, `RemoteCacheManager`, `RedisRemoteCacheBackend`). Factory helpers and key helpers. | `runtime/cache.py` |
| knobs | Process-global mutable configuration, hook chains, and callback slots. Module-level singletons for `build`, `cache`, `compilation`, `runtime`, `language`, `nvidia`, `amd`, `proton`. | `knobs.py` |
| backends | Backend discovery and abstract interfaces. Global `backends` dict populated at import time. `BaseBackend`, `GPUTarget`, `DriverBase` abstractions. | `backends/__init__.py`, `backends/compiler.py`, `backends/driver.py` |
| ir | MLIR IR construction bindings. `MLIRContext`, dialect registration, operation builders, pass manager. GIL released on long-running MLIR ops via `call_guard`. | `python/src/ir.cc` |
| llvm | LLVM compilation bindings. Module-level optimization, assembly/object emission. Multiple `py::gil_scoped_release` points where C++ runs concurrently with Python threads. | `python/src/llvm.cc` |
| src-main | Module entry point for `triton._C.libtriton`. Submodule wiring and initialization. | `python/src/main.cc` |
| src-passes | MLIR pass-pipeline builder functions and analysis wrappers (`ModuleAllocation`, `ModuleMembarAnalysis`). | `python/src/passes.cc` |
| runtime-build | Dynamic C module compilation for launcher stubs and GPU memory allocator interface. | `runtime/build.py`, `runtime/_allocation.py` |
| interpreter | CPU interpreter fallback for Triton kernels and native interpreter helpers for atomic operations. | `runtime/interpreter.py`, `python/src/interpreter.cc` |
| compiler-codegen | Python AST to MLIR IR translation and launcher code generation. Front-end lowering and C launcher generation. | `compiler/code_generator.py`, `compiler/make_launcher.py` |
| native-helpers | Lower-frequency native bindings for LinearLayout utilities and Gluon IR operations. | `python/src/linear_layout.cc`, `python/src/gluon_ir.cc` |
| language | DSL operations (`tl.load`, `tl.store`, etc.), type system, and language primitives. Constexpr and declarative operators. | `language/core.py` |
| tools | Offline compilation, linking, and disassembly utilities. CLI entry points for Triton tooling. | `tools/compile.py`, `tools/link.py`, `tools/disasm.py` |
| experimental | Experimental Gluon features and utilities. | `experimental/gluon/` |


## Concurrency Model

Triton's primary concurrency exposure comes from JIT compilation and kernel dispatch. We classify issues by the use case they affect.

**This is the canonical tier definition.** Other docs in this repo refer back here.

### Tier 1 — Floor: Triton can be used at all from a multi-threaded program

One thread uses Triton; other threads in the same process do unrelated Python
work (data loaders, request handlers, async dispatch loops).

What must be true: concurrent first-time entry into Triton's lazy
initialization paths is safe (no TOCTOU on driver discovery, native
specialization helpers, or module-level caches), and Triton activity in one
thread cannot corrupt unrelated threads.

Typical risks:
- global registries (`_triton_jit_function_registry`)
- lazy singleton initialization (`DriverConfig`, `CudaUtils`)
- first-use initialization of process-global C++ state (`init_globals` in `specialize.cc`)
- native helper code that assumed the GIL serialized access

This is what every nogil-aware library must ship.

### Tier 2 — Concurrent dispatch to distinct streams / devices

Multiple threads reuse the same compiled `JITFunction` / `Autotuner` to launch
kernels concurrently, typically each thread owning a different CUDA stream
and/or GPU. Multiple threads driving a single stream is **not** a goal — kernel
ordering on one stream is the caller's problem.

Typical risks:
- per-instance caches on `JITFunction` and `Autotuner` (`device_caches`, `kernel_cache`, `Autotuner.cache`)
- lazy finalization of `CompiledKernel` runtime handles
- races in the C++ specialization helper (`dtype_ptr2str`, `dtype2str`, `type_handler_cache`)
- launch-path hook chains read while being mutated

### Tier 3 — Concurrent compilation and concurrent configuration mutation

A strict superset of Tier 2. Multiple threads can compile different kernels
concurrently, *and* threads can mutate Triton configuration (knobs, hooks,
driver/backend selection) while other threads compile or launch — all without
the caller having to know which mutations are safe in which window.

Typical risks:
- Python-level state on the compile orchestration path that the GIL was
  implicitly serializing (`code_generator` gscope iteration, kernel-load hook
  chains, `add_stages_inspection_hook` TOCTOU, `used_global_vals` reads).
- `MLIRContext` sharing across threads (open question — see below).
- Mutation of `knobs.*` flags during execution.
- Hook chain add/remove during execution.
- Driver / backend selection changing while another thread compiles or launches.

Note on the GIL build today: the Triton maintainers report that multi-threaded
compilation is *already used in production*, but it works only because callers
follow rules like "don't mutate LLVM `cl::opt` per-kernel" and "don't change
knobs while kernels are compiling/launching." Real Tier 3 support would not
require users to know those rules — that's the difference between "works if
you're careful" and "works."

Open question to confirm with maintainers: is `MLIRContext` per-compile or
shared across threads? Multi-threaded compile works under GIL, so either it's
per-compile or shared use is already protected. The answer determines the size
of the `ir/` and `native-helpers/` Tier 3 fix list.

### Pre-existing rule violations (not free-threading bugs)

Some issues uncovered during the audit are violations of pre-existing
maintainer rules, independent of free-threading. They should be filed
separately rather than as nogil bugs:

- **Mutating LLVM `cl::opt` from runtime code.** The maintainers' rule is
  "don't do that"; only debug features currently do it. Real Tier 3 support
  implies removing this mutation, but the bug exists under the GIL build too.

### Pragmatically out of scope

- **Concurrent use of `TRITON_INTERPRET=1` interpreter mode.** Debug-only
  fallback path that monkey-patches shared module state. Pragmatic fix is a
  single process-level lock around `GridExecutor.__call__`; document as
  single-threaded.

### Tier classification when auditing

When auditing, note which tier an issue falls into. Tiers are cumulative:
shipping Tier 2 implies Tier 1 already holds, and shipping Tier 3 implies Tier
2 already holds. An issue's tier is the lowest use case that the issue breaks.

## Components

### Runtime (`python/triton/runtime/`)

| Component | Key files | Description |
|-----------|-----------|-------------|
| **jit** | `jit.py` | `@triton.jit` decorator, `JITFunction` class, kernel dispatch, specialization. Global `_triton_jit_function_registry` dict; per-instance `device_caches` defaultdict; `@cached_property` on `KernelParam`. |
| **autotuner** | `autotuner.py` | `@autotune` decorator, config benchmarking. Per-instance `cache` and `configs_timings` dicts; disk cache read/write. |
| **driver** | `driver.py` | `DriverConfig` singleton managing active GPU driver. Lazy-init `_default`/`_active` properties without synchronization. |
| **cache** | `cache.py` | `FileCacheManager` (on-disk) and `RemoteCacheManager` (Redis). LRU-cached `triton_key()`. Redis backend holds a singleton connection. |
| **async-compile** | `_async_compile.py` | `AsyncCompileMode` context manager for background compilation via `concurrent.futures`. Uses `ContextVar` (safe), but `FutureKernel` result caching needs review. |
| **allocation** | `_allocation.py` | GPU memory allocator interface. `ContextVar`-based (safe). Mutable `_profile_allocator` wrapper singleton. |
| **build** | `build.py` | Dynamic C module compilation for launcher stubs. LRU-cached `platform_key()`. |
| **interpreter** | `interpreter.py` | CPU interpreter fallback (no GPU). Module-level cached numpy ufuncs. |

### Compiler (`python/triton/compiler/`)

| Component | Key files | Description |
|-----------|-----------|-------------|
| **compiler** | `compiler.py` | Main compilation pipeline orchestration. LRU-cached `max_shared_mem()`; `CompileTimer` accumulates timing data; `LazyDict`/`AsmDict` mutable compilation artifacts. |
| **code-generator** | `code_generator.py` | Python AST to MLIR IR translation. Per-compilation instance state (`local_defs`, `lscope`). |
| **make-launcher** | `make_launcher.py` | Generates C launcher source code. Minimal state. |

### Knobs & Configuration

| Component | Key files | Description |
|-----------|-----------|-------------|
| **knobs** | `knobs.py` | Global config singletons (`build`, `cache`, `compilation`, `runtime`, `language`, `nvidia`, `amd`, `proton`). Env-var-backed descriptor protocol; mutable `__dict__`; `HookChain` callback lists; `scope()` context manager saves/restores state. |

### Backends (`python/triton/backends/`)

| Component | Key files | Description |
|-----------|-----------|-------------|
| **backends** | `__init__.py`, `compiler.py`, `driver.py` | Backend discovery and abstract interfaces. Global `backends` dict populated at import time. `GPUDriver` caches torch CUDA functions at init. |

### Language (`python/triton/language/`)

| Component | Key files | Description |
|-----------|-----------|-------------|
| **language** | `core.py`, `math.py`, `semantic.py`, `standard.py` | DSL operations (`tl.load`, `tl.store`, etc.), type system, constexpr. Mostly declarative/immutable. Global `env` namespace object. |

### Tools (`python/triton/tools/`)

| Component | Key files | Description |
|-----------|-----------|-------------|
| **tools** | `compile.py`, `link.py`, `disasm.py` | Offline compilation, linking, and disassembly utilities. Primarily CLI entry points; lower concurrency exposure. |

### C++/pybind11 (`python/src/`)

| Component | Key files | Description |
|-----------|-----------|-------------|
| **specialize** | `specialize.cc` | Kernel argument type specialization. Static global pointers (`constexpr_cls`, etc.), static `dtype_ptr2str`/`dtype2str`/`type_handler_cache` maps; one-shot `init_globals()` without locking. |
| **ir** | `ir.cc` | MLIR IR construction bindings. Static `llvm::raw_fd_ostream` singleton (not thread-safe). GIL released on long-running MLIR ops via `call_guard`. |
| **llvm** | `llvm.cc` | LLVM compilation bindings. Multiple `py::gil_scoped_release` points where C++ runs concurrently with Python threads. |
| **passes** | `passes.cc` | MLIR transformation pass bindings. |
| **interpreter** | `interpreter.cc` | Interpreter-mode atomic ops. Protected by `std::mutex atomic_op_guard`. |
| **linear-layout** | `linear_layout.cc` | LinearLayout bindings. |
