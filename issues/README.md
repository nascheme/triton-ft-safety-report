# Triton Free-Threading Issue Index

This directory tracks thread-safety findings in Triton's Python layer for
Python 3.14t (free-threading / nogil).

The top-level project context lives in [`../README.md`](../README.md). See
[`../CLAUDE.md`](../CLAUDE.md) for the audit methodology and
[`../OVERVIEW.md`](../OVERVIEW.md) for the codebase overview.

## Proposed Patches

Each issue below has a patch file along with an issue file
(`issues/<component>/<issue>.patch` from the repo root). This is a patch queue,
not a count of every Tier 1-2 issue ID: some patch entries cover multiple issue
IDs, and some issue IDs are tracked as upstream symptoms or as consequences of
another patch. See [Detailed Issue Counts](#detailed-issue-counts) for the
canonical counts.

### Severity: HIGH

- FT044: [`specialize/init-globals-toctou`](specialize/init-globals-toctou.md) (Tier 1) -
  A plain-bool guard lets two threads run `init_globals()` at once, racing
  file-scope `PyObject*` assignments and `type_handler_cache`.
- FT002: [`autotuner/cache-toctou`](autotuner/cache-toctou.md) (Tier 2) -
  Concurrent `run()` calls both miss the config cache and benchmark in
  parallel, racing shared scratch state: wrong-tensor restores, disk cache
  keyed wrong.
- FT010: [`compiler/compiled-kernel-init-handles-race`](compiler/compiled-kernel-init-handles-race.md) (Tier 2) -
  `_init_handles` publishes `_run` before `module`/`function` are set; a second
  thread skips init and launches with a null function handle.
- FT020: [`jit/device-caches-race`](jit/device-caches-race.md) (Tier 2) -
  Non-atomic `defaultdict` fill of `device_caches` lets two threads build
  separate cache tuples; the loser uses orphaned caches and silently
  recompiles.
- FT042/FT043: [`specialize/dtype-ptr2str-unordered-map-race`](specialize/dtype-ptr2str-unordered-map-race.md) /
  [`specialize/dtype2str-unordered-map-race`](specialize/dtype2str-unordered-map-race.md) (Tier 2) -
  `handle_tensor()` and `specialize_tensordesc()` each mutate a process-global
  `unordered_map` (`dtype_ptr2str` / `dtype2str`) with no lock, risking
  container corruption under concurrent launches. Fixed by a single patch.

### Severity: MED

- FT009: [`compiler/code-generator-gscope-iteration-race`](compiler/code-generator-gscope-iteration-race.md) (Tier 1) -
  `get_capture_scope()` returns the module `__dict__` by identity and the
  compiler iterates it; a concurrent global rebind yields a skipped global
  (`NameError`) or a stale-keyed kernel.
- FT021: [`jit/function-registry-race`](jit/function-registry-race.md) (Tier 1) -
  `__init__` publishes `self` to the global registry before populating
  attributes; another thread sees a half-built `JITFunction` and raises
  `AttributeError`.
- FT040: [`runtime-driver/driver-default-lazy-init-race`](runtime-driver/driver-default-lazy-init-race.md) (Tier 1) -
  `DriverConfig.default`/`active` do unlocked check-then-assign; two threads
  build duplicate backend drivers and one is leaked (likely root of
  triton#6721).
- FT013: [`experimental/symmetric-memory-rendezvous-toctou`](experimental/symmetric-memory-rendezvous-toctou.md) (Tier 2) -
  `rendezvous()` check-then-act on `_RENDEZVOUS_CACHE` lets both threads run
  the full protocol: double-freed handle, duplicate runtime-state import.
- FT019: [`jit/async-compile-races`](jit/async-compile-races.md) (Tier 2) -
  Unlocked `AsyncCompileMode.submit` and `FutureKernel` finalize defeat
  memoization and fire post-compile hooks twice per compile.
- FT023: [`jit/kernel-cache-toctou`](jit/kernel-cache-toctou.md) (Tier 2) -
  Lock-free `kernel_cache` get/compile/set in `run()` lets two threads recompile
  the same kernel; the async path can leave a stale `FutureKernel`.
- FT025: [`jit/unsafe-update-src-race`](jit/unsafe-update-src-race.md) (Tier 2) -
  `_unsafe_update_src` writes `_src`/`hash` without `_hash_lock`; a lost
  invalidation makes later compiles reuse the old source's cache entry.
- FT026: [`jit/used-global-vals-unsynchronized-read`](jit/used-global-vals-unsynchronized-read.md) (Tier 2) -
  Unordered `kernel_cache` and `used_global_vals` writes let a cache hit observe
  empty `used_global_vals` and skip the global-changed safety check.

## Detailed Issue Counts

Counts below are generated from the detailed issue files under
`issues/<component>/`. Component-summary-only candidates and rejected notes are
not counted here.

There are **21 Tier 1-2 HIGH/MED issue IDs**: **8 HIGH** and **13 MED**.
Tier 3 findings are tracked for audit completeness, but they are not part of
the current Tier 1-2 free-threading support goal.

| Component | HIGH | MED | LOW | Other | Detailed issue files | Tier 1-2 HIGH/MED |
|-----------|-----:|----:|----:|------:|---------------------:|------------------:|
| [autotuner](autotuner.md) | 1 | 4 | 2 | 0 | 7 | 5 |
| [compiler](compiler.md) | 1 | 3 | 0 | 0 | 4 | 2 |
| [experimental](experimental.md) | 0 | 2 | 0 | 0 | 2 | 2 |
| [interpreter](interpreter.md) | 2 | 2 | 0 | 0 | 4 | 0 |
| [jit](jit.md) | 1 | 7 | 0 | 1 | 9 | 6 |
| [knobs](knobs.md) | 0 | 1 | 0 | 0 | 1 | 0 |
| [llvm](llvm.md) | 0 | 0 | 3 | 0 | 3 | 0 |
| [native-helpers](native-helpers.md) | 0 | 0 | 1 | 4 | 5 | 0 |
| [nvidia-driver](nvidia-driver.md) | 0 | 0 | 2 | 0 | 2 | 0 |
| [runtime-driver](runtime-driver.md) | 1 | 3 | 0 | 0 | 4 | 2 |
| [specialize](specialize.md) | 4 | 0 | 0 | 0 | 4 | 4 |
| **Total** | **10** | **22** | **8** | **5** | **45** | **21** |

## Tier Model

Triton's primary concurrency exposure comes from JIT compilation and kernel
dispatch. We classify issues by the use case they affect.

**This is the canonical tier definition.** Other docs in this repo refer back
here.

### Why Free-Threaded Triton Matters

PyTorch, which sits on top of Triton, has concrete free-threading use cases that
motivate this audit. In rough order of importance:

- **GPU multiplexing.** Multiple threads share one GPU, alternating CPU-bound
  and GPU-bound work to keep the GPU busy. Reported in production workloads;
  see [pytorch/pytorch#180550](https://github.com/pytorch/pytorch/issues/180550).
- **One process, N GPUs, one thread per GPU.** Process-per-GPU is the more
  common deployment today, but single-process is easier to debug and makes
  state-sharing across GPUs trivial.
- **One GPU-compute thread plus N data-loader threads**, where data loaders may
  touch the GPU lightly.

Research workloads exhibit more variety than production / high-effort training,
and free-threading lowers the cost of expressing those programs in threads
instead of processes.

### Tier 1 - Floor: Triton Can Be Used At All From A Multi-Threaded Program

Exactly one thread uses Triton; other threads in the same process do unrelated
Python work. The canonical example is one GPU-compute thread plus N data-loader
threads, as long as the loader threads do not themselves call into Triton. Light
GPU use from a non-Triton thread (CUDA copies, non-Triton torch ops) is still
Tier 1; the moment a second thread launches a Triton kernel, the program is
Tier 2.

What must be true: concurrent first-time entry into Triton's lazy initialization
paths is safe (no TOCTOU on driver discovery, native specialization helpers, or
module-level caches), and Triton activity in one thread cannot corrupt unrelated
threads.

Typical risks:

- concurrent first use of global registries
- lazy singleton initialization
- first-use initialization of process-global C++ state
- native helper code that assumed the GIL serialized access

### Tier 2 - Concurrent Use Of Triton From Multiple Threads

Multiple threads in the same process use Triton concurrently. The two motivating
use cases are GPU multiplexing on a single GPU and one-thread-per-GPU across N
GPUs. Both compiled-kernel dispatch and first-time kernel compilation are in
scope: multi-threaded compilation is already a production use case under the
GIL-enabled build and must keep working under the free-threaded build.

The typical pattern is one CUDA stream per thread. Multiple threads sharing a
single stream is also correct: it just adds extra kernel-ordering dependencies
that the caller has chosen to accept; it is not excluded.

Typical risks:

- per-instance caches on shared wrapper objects
- lazy finalization of `CompiledKernel` runtime handles
- races in the C++ specialization helper
- Python-level state on the compile path that the GIL was implicitly
  serializing
- launch-path hook chains read under concurrent reads
- async-compile bookkeeping

Per-component issue files carry the concrete findings and the reasoning behind
each severity/tier call.

### Tier 3 - Unsupported Concurrent Scenarios (Tracked, Not A Goal)

Tier 3 tracks scenarios that are useful to record for audit completeness but
are not part of the current free-threading support objective. These are
generally scenarios already unsupported or not positioned for concurrent use in
the GIL-enabled build.

The main Tier 3 families are:

- concurrent mutation of Triton configuration while other threads compile or
  launch kernels
- `HookChain.add` / `HookChain.remove` and hook-slot reassignment during
  execution
- `base_knobs.scope()` context manager use across threads
- driver / backend selection changing while another thread compiles or launches
- concurrent use of `TRITON_INTERPRET=1` interpreter mode
- any other unsupported "configuration changes or debug-mode execution
  mid-flight" race

**Concurrent interpreter mode:** The `TRITON_INTERPRET=1` path is a debug-only
fallback that monkey-patches shared module state. The pragmatic current
position is to document it as single-threaded or serialize
`GridExecutor.__call__` with one process-level lock. Interpreter findings are
therefore Tier 3 today, but the HIGH/MED findings in
[`interpreter.md`](interpreter.md) are serious correctness issues if concurrent
interpreter execution becomes an in-scope support goal.

### Pre-Existing Rule Violations

Issues uncovered during the audit that are violations of pre-existing
maintainer rules, independent of free-threading, are filed separately rather
than as free-threaded specific bugs.

- **Mutating LLVM `cl::opt` from runtime code.** The maintainers' rule is
  "don't do that"; only debug features currently do it. The bug exists under
  the GIL-enabled build too. Tier 3 issues belong to the same family: scenarios
  that are already unsupported under the GIL-enabled build, and we do not
  propose to support them under free-threading either.

### Tier Classification When Auditing

When auditing, note which tier an issue falls into. Tiers are support goals:
shipping Tier 2 implies Tier 1 already holds. Tier 3 is tracked separately as
unsupported behavior, even when its internal race shape resembles a Tier 2
multi-threaded execution race.

- If an issue only manifests when configuration is mutated concurrently with
  execution (knobs, hook chains, driver/backend), it is **Tier 3**: tracked but
  not a current goal.
- If an issue only manifests under concurrent `TRITON_INTERPRET=1` execution,
  it is **Tier 3**: serious if interpreter concurrency becomes in-scope, but
  outside the current Tier 1-2 objective.
- If an issue manifests during concurrent compilation of different kernels with
  no concurrent configuration mutation or debug-only interpreter mode, it is
  **Tier 2**.
- If an issue manifests when one thread uses Triton while non-Triton Python
  threads do unrelated work, it is **Tier 1**.
