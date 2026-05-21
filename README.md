# Triton Free-Threading Safety Audit

Tracking thread-safety issues in Triton's Python layer for Python 3.14t (free-threading / nogil).

We are auditing the Python code under `python/` (both the pure-Python `triton/`
package and the C++ pybind11 extensions in `src/`). The C++/MLIR/LLVM core is
out of scope except where it interacts with the Python layer (GIL release
points, pybind11 bindings).

See [CLAUDE.md](CLAUDE.md) for the audit methodology.
See [OVERVIEW.md](OVERVIEW.md) for an architectural overview of the Triton
codebase, including a description of each audit component.


## Audit Progress

| Component | Status | SEVERE | Significant | Minor | Total |
|-----------|--------|--------|-------------|-------|-------|
| [jit](issues/jit.md) | In-review | 1 | 7 | 0 | 8 |
| [autotuner](issues/autotuner.md) | In-review | 1 | 4 | 2 | 7 |
| [compiler](issues/compiler.md) | In-review | 1 | 3 | 0 | 4 |
| [runtime-driver](issues/runtime-driver.md) | In-review | 1 | 3 | 0 | 4 |
| [nvidia-driver](issues/nvidia-driver.md) | In-review | 0 | 0 | 3 | 3 |
| [specialize](issues/specialize.md) | In-review | 4 | 1 | 1 | 6 |
| [async-compile](issues/async-compile.md) | In-review | 0 | 2 | 2 | 4 |
| [cache](issues/cache.md) | In-review | 0 | 0 | 4 | 4 |
| [knobs](issues/knobs.md) | In-review | 1 | 4 | 3 | 8 |
| [backends](issues/backends.md) | In-review | 0 | 0 | 2 | 2 |
| [ir](issues/ir.md) | In-review | 0 | 1 | 7 | 8 |
| [llvm](issues/llvm.md) | In-review | 0 | 0 | 8 | 8 |
| [src-main](issues/src-main.md) | In-review | 0 | 0 | 1 | 1 |
| [src-passes](issues/src-passes.md) | In-review | 0 | 0 | 4 | 4 |
| [runtime-build](issues/runtime-build.md) | In-review | 0 | 1 | 5 | 6 |
| [interpreter](issues/interpreter.md) | In-review | 2 | 2 | 3 | 7 |
| [compiler-codegen](issues/compiler-codegen.md) | In-review | 0 | 0 | 0 | 0 |
| [native-helpers](issues/native-helpers.md) | In-review | 1 | 3 | 1 | 5 |
| [language](issues/language.md) | In-review | 0 | 0 | 6 | 6 |
| [tools](issues/tools.md) | In-review | 0 | 0 | 4 | 4 |
| [experimental](issues/experimental.md) | In-review | 0 | 3 | 4 | 7 |
| **Total** | | **12** | **34** | **60** | **106** |

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
- concurrent first use of global registries
- lazy singleton initialization
- first-use initialization of process-global C++ state
- native helper code that assumed the GIL serialized access


### Tier 2 — Concurrent use of Triton from multiple threads

Multiple threads in the same process use Triton concurrently. The two
motivating use cases are GPU multiplexing on a single GPU and one-thread-per-
GPU across N GPUs. Both compiled-kernel dispatch and first-time kernel
compilation are in scope: multi-threaded compilation is already a production
use case under the GIL-enabled build and must keep working under the
free-threaded build.

The typical pattern is one CUDA stream per thread. Multiple threads sharing a
single stream is also correct — it just adds extra kernel-ordering
dependencies that the caller has chosen to accept; it is not excluded.

Typical risks:
- per-instance caches on shared wrapper objects
- lazy finalization of `CompiledKernel` runtime handles
- races in the C++ specialization helper
- Python-level state on the compile path that the GIL was implicitly
  serializing
- launch-path hook chains read under concurrent reads
- async-compile bookkeeping

Per-component issue files (`issues/<component>.md`) carry the concrete
findings and the reasoning behind each severity/tier call.

### Tier 3 — Concurrent configuration mutation (tracked, not a goal)

A strict superset of Tier 2 in which threads can mutate Triton configuration
(knobs, hook chains, driver/backend selection) while other threads compile
or launch.

**Tier 3 is tracked for audit completeness, not a current objective.** We are
not asking Triton maintainers to make these scenarios safe under
free-threading. The underlying scenarios are already unsupported under the
GIL-enabled build today:

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
than as free-threaded specific bugs:

- **Mutating LLVM `cl::opt` from runtime code.** The maintainers' rule is
  "don't do that"; only debug features currently do it. The bug exists
  under the GIL-enabled build too. Tier 3 issues belong to the same family — they
  are scenarios that are already unsupported under the GIL-enabled build, and we
  do not propose to support them under free-threading either.

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
