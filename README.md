# Triton Free-Threading Safety Audit

Tracking thread-safety issues in Triton's Python layer for Python 3.14t (free-threading / nogil).

We are auditing the Python code under `python/` (both the pure-Python `triton/` package and the C++ pybind11 extensions in `src/`). The C++/MLIR/LLVM core is out of scope except where it interacts with the Python layer (GIL release points, pybind11 bindings).

See [CLAUDE.md](CLAUDE.md) for the audit methodology.
See [OVERVIEW.md](OVERVIEW.md) for an architectural overview of the Triton codebase.

## Concurrency Model

Triton's primary concurrency exposure comes from JIT compilation and kernel dispatch:

- **Tier 1 (urgent):** A single thread compiles and launches kernels while other threads do data loading or other Python work. The main risk is races on global registries (`_triton_jit_function_registry`), lazy-init singletons (`DriverConfig`), and `@cached_property` on shared objects.
- **Tier 2 (goal):** Multiple threads compile and launch Triton kernels concurrently, exposing races on `JITFunction.device_caches`, `Autotuner.cache`, and the global `knobs.*` singletons.

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

## Audit Progress

| Component | Status | Report |
|-----------|--------|--------|
| (audit not yet started) | | |
