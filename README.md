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
| [jit](issues/jit.md) | In-review | 1 | 7 | 0 | 8 |
| [autotuner](issues/autotuner.md) | In-review | 1 | 4 | 2 | 7 |
| [compiler](issues/compiler.md) | In-review | 1 | 3 | 0 | 4 |
| [runtime-driver](issues/runtime-driver.md) | In-review | 1 | 3 | 0 | 4 |
| [nvidia-driver](issues/nvidia-driver.md) | In-review | 0 | 0 | 2 | 2 |
| [specialize](issues/specialize.md) | Ongoing | 4 | 1 | 1 | 6 |
| [async-compile](issues/async-compile.md) | Ongoing | 0 | 2 | 2 | 4 |
| [cache](issues/cache.md) | Ongoing | 0 | 0 | 4 | 4 |
| [knobs](issues/knobs.md) | Ongoing | 1 | 4 | 3 | 8 |
| [backends](issues/backends.md) | Ongoing | 0 | 0 | 2 | 2 |
| [ir](issues/ir.md) | Ongoing | 4 | 3 | 1 | 8 |
| [llvm](issues/llvm.md) | Ongoing | 2 | 5 | 1 | 8 |
| [src-main](issues/src-main.md) | Ongoing | 0 | 0 | 1 | 1 |
| [src-passes](issues/src-passes.md) | Ongoing | 0 | 2 | 2 | 4 |
| [runtime-build](issues/runtime-build.md) | Ongoing | 0 | 1 | 5 | 6 |
| [interpreter](issues/interpreter.md) | Ongoing | 2 | 2 | 3 | 7 |
| [compiler-codegen](issues/compiler-codegen.md) | Ongoing | 0 | 0 | 0 | 0 |
| [native-helpers](issues/native-helpers.md) | Ongoing | 1 | 3 | 1 | 5 |
| [language](issues/language.md) | Ongoing | 0 | 0 | 6 | 6 |
| [tools](issues/tools.md) | Ongoing | 0 | 0 | 4 | 4 |
| [experimental](issues/experimental.md) | Ongoing | 0 | 3 | 4 | 7 |
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

Triton's primary concurrency exposure comes from JIT compilation and kernel
dispatch. We classify issues by the use case they affect.

**This is the canonical tier definition.** Other docs in this repo refer back
here.

### Why free-threaded Triton matters

PyTorch — which sits on top of Triton — has concrete free-threading use cases
that motivate this audit. In rough order of importance:

- **GPU multiplexing.** Multiple threads share one GPU, alternating CPU-bound
  and GPU-bound work to keep the GPU busy. Reported in production workloads;
  see [pytorch/pytorch#180550](https://github.com/pytorch/pytorch/issues/180550).
- **One process, N GPUs, one thread per GPU.** Process-per-GPU is the more
  common deployment today, but single-process is easier to debug and makes
  state-sharing across GPUs trivial.
- **One GPU-compute thread plus N data-loader threads**, where data loaders
  may touch the GPU lightly.

Research workloads exhibit more variety than production / high-effort
training, and free-threading lowers the cost of expressing those programs in
threads instead of processes.

### Tier 1 — Floor: Triton can be used at all from a multi-threaded program

Exactly one thread uses Triton; other threads in the same process do unrelated
Python work. The canonical example is one GPU-compute thread plus N
data-loader threads, *as long as the loader threads do not themselves call
into Triton*. Light GPU use from a non-Triton thread (CUDA copies, non-Triton
torch ops) is still Tier 1; the moment a second thread launches a Triton
kernel, the program is Tier 2.

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

### Tier 2 — Concurrent use of Triton from multiple threads

Multiple threads in the same process use Triton concurrently. The two
motivating use cases are GPU multiplexing on a single GPU and one-thread-per-
GPU across N GPUs. Both compiled-kernel dispatch and first-time kernel
compilation are in scope: multi-threaded compilation is already a production
use case under the GIL build and must keep working under nogil.

The typical pattern is one CUDA stream per thread. Multiple threads sharing a
single stream is also correct — it just adds extra kernel-ordering
dependencies that the caller has chosen to accept; it is not excluded.

Typical risks:
- per-instance caches on shared wrapper objects (`JITFunction.device_caches`,
  `JITFunction.kernel_cache`, `Autotuner.cache`)
- lazy finalization of `CompiledKernel` runtime handles
- races in the C++ specialization helper (`dtype_ptr2str`, `dtype2str`,
  `type_handler_cache`)
- Python-level state on the compile path that the GIL was implicitly
  serializing (`code_generator` gscope iteration, `JITCallable.used_global_vals`,
  source-mutation paths consulted during compile)
- launch-path hook chains *read* under concurrent reads
- async-compile bookkeeping (`FutureKernel` result caching)

Open question to confirm with maintainers: is `MLIRContext` per-compile or
shared across threads? Multi-threaded compile already works under the GIL,
so either it is per-compile or shared use is already protected. The answer
determines the size of the `ir/` and `native-helpers/` Tier 2 fix list.

### Tier 3 — Concurrent configuration mutation (tracked, not a goal)

A strict superset of Tier 2 in which threads can mutate Triton configuration
(knobs, hook chains, driver/backend selection) while other threads compile
or launch.

**Tier 3 is tracked for audit completeness, not a current objective.** We are
not asking Triton maintainers to make these scenarios safe under nogil. The
underlying scenarios are already unsupported under the GIL build today:

- The maintainers' rule on LLVM `cl::opt` is "don't change it per-kernel";
  only debug features currently do it.
- The `base_knobs.scope()` context manager is documented (and confirmed by
  maintainers) as unsafe to use across threads even today.
- The general expectation is that `knobs.*` and hook configuration are set
  before kernels start compiling or launching, not mutated mid-flight.

Issues tracked under Tier 3:
- mutation of `knobs.*` attributes while kernels are compiling or launching
- `HookChain.add` / `HookChain.remove` and hook-slot reassignment during
  execution
- `base_knobs.scope()` context manager
- driver / backend selection changing while another thread compiles or
  launches
- any other "configuration changes mid-execution" race

### Pre-existing rule violations (not free-threading bugs)

Issues uncovered during the audit that are violations of pre-existing
maintainer rules — independent of free-threading. Filed separately rather
than as nogil bugs:

- **Mutating LLVM `cl::opt` from runtime code.** The maintainers' rule is
  "don't do that"; only debug features currently do it. The bug exists
  under the GIL build too. Tier 3 issues belong to the same family — they
  are scenarios that are already unsupported under the GIL build, and we
  do not propose to support them under nogil either.

### Pragmatically out of scope

- **Concurrent use of `TRITON_INTERPRET=1` interpreter mode.** Debug-only
  fallback path that monkey-patches shared module state. Pragmatic fix is a
  single process-level lock around `GridExecutor.__call__`; document as
  single-threaded.

### Tier classification when auditing

When auditing, note which tier an issue falls into. Tiers are cumulative:
shipping Tier 2 implies Tier 1 already holds, and shipping Tier 3 implies
Tier 2 already holds. An issue's tier is the lowest use case that the issue
breaks.

- If an issue only manifests when configuration is mutated concurrently with
  execution (knobs, hook chains, driver/backend), it is **Tier 3** —
  tracked but not a current goal.
- If an issue manifests during concurrent compilation of different kernels
  with no concurrent configuration mutation, it is **Tier 2**.

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
