# Triton Free-Threading Issue Index

This directory tracks thread-safety findings in Triton's Python layer for
Python 3.14t (free-threading / nogil).

The top-level project context lives in [`../README.md`](../README.md). See
[`../CLAUDE.md`](../CLAUDE.md) for the audit methodology and
[`../OVERVIEW.md`](../OVERVIEW.md) for the codebase overview.

## Rank Model

The report uses one triage axis. Readers should not need to combine a risk
column with a support-scope column to decide what matters.

| Rank | Meaning |
| --- | --- |
| **Critical Blocker** | Highest-priority current-goal blocker: plausible crash, memory/container corruption, wrong-kernel execution, or corrupted runtime/compiler state. |
| **Blocker** | Must be fixed for the current free-threaded Triton goal, but with a less catastrophic or more recoverable consequence than Critical Blocker. |
| **Deferred** | Serious race shape, but it requires unsupported concurrent configuration mutation, debug-mode execution, driver/backend switching, or `TRITON_INTERPRET=1`. Track it, but do not block the current goal on it. |
| **Low** | Low-impact, latent, diagnostic-only, duplicate-work, or cold-start behavior. |
| **Rejected** | Audited and not considered a free-threading bug or Triton-owned fix. |
| **Out-of-scope** | Real concern, but outside this free-threading report, usually because it violates a pre-existing maintainer rule independent of no-GIL Python. |

`TRITON_INTERPRET=1` findings are **Deferred** today. They are serious once
concurrent interpreter execution becomes an in-scope support goal, but the
interpreter is currently a debug-only single-threaded path.

## Patch Queue

Each issue below has a patch file along with an issue file
(`issues/<component>/<issue>.patch` from the repo root). This is a patch queue,
not a count of every Blocker issue ID: some patch entries cover multiple issue
IDs, and some issue IDs are tracked as upstream symptoms or as consequences of
another patch. See [Rank Counts](#rank-counts) for the canonical counts.

Entries in this queue are ranked **Critical Blocker** or **Blocker**.

| Issue | Rank | Patch / risk |
| --- | --- | --- |
| FT044 | Critical Blocker | [`specialize/init-globals-toctou`](specialize/init-globals-toctou.md): plain-bool lazy init lets two threads race file-scope `PyObject*` assignment and `type_handler_cache`. |
| FT002 | Critical Blocker | [`autotuner/cache-toctou`](autotuner/cache-toctou.md): concurrent cache misses benchmark in parallel and race shared autotuner scratch state. |
| FT010 | Critical Blocker | [`compiler/compiled-kernel-init-handles-race`](compiler/compiled-kernel-init-handles-race.md): `_run` can be published before `module` / `function` are initialized. |
| FT020 | Critical Blocker | [`jit/device-caches-race`](jit/device-caches-race.md): `device_caches` can build orphaned cache tuples under concurrent access. |
| FT042/FT043 | Critical Blocker | [`specialize/dtype-ptr2str-unordered-map-race`](specialize/dtype-ptr2str-unordered-map-race.md) / [`specialize/dtype2str-unordered-map-race`](specialize/dtype2str-unordered-map-race.md): global C++ `unordered_map` caches mutate without synchronization. |
| FT009 | Blocker | [`compiler/code-generator-gscope-iteration-race`](compiler/code-generator-gscope-iteration-race.md): compiler iterates a live module `__dict__`, so concurrent global rebinding can miscompile or raise. |
| FT021 | Blocker | [`jit/function-registry-race`](jit/function-registry-race.md): global registry can expose a half-built `JITFunction`. |
| FT040 | Blocker | [`runtime-driver/driver-default-lazy-init-race`](runtime-driver/driver-default-lazy-init-race.md): unlocked lazy driver initialization can duplicate backend drivers. |
| FT013 | Blocker | [`experimental/symmetric-memory-rendezvous-toctou`](experimental/symmetric-memory-rendezvous-toctou.md): rendezvous cache check-then-act can run the protocol twice. |
| FT019 | Blocker | [`jit/async-compile-races`](jit/async-compile-races.md): async compile bookkeeping can defeat memoization and run finalization hooks twice. |
| FT023 | Blocker | [`jit/kernel-cache-toctou`](jit/kernel-cache-toctou.md): lock-free compile cache path can duplicate compilation and leave stale futures. |
| FT025 | Blocker | [`jit/unsafe-update-src-race`](jit/unsafe-update-src-race.md): source/hash invalidation is not synchronized with cache-key computation. |
| FT026 | Blocker | [`jit/used-global-vals-unsynchronized-read`](jit/used-global-vals-unsynchronized-read.md): cache hits can skip the global-changed safety check. |

## Rank Counts

Counts below are generated from the detailed issue files under
`issues/<component>/`. Component-summary-only candidates and rejected notes are
not counted here.

There are **21 current-goal issue IDs**: **8 Critical Blocker** and
**13 Blocker**.

| Component | Critical Blocker | Blocker | Deferred | Low | Rejected | Out-of-scope | Detailed issue files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [autotuner](autotuner.md) | 1 | 4 | 0 | 2 | 0 | 0 | 7 |
| [compiler](compiler.md) | 1 | 1 | 2 | 0 | 0 | 0 | 4 |
| [experimental](experimental.md) | 0 | 2 | 0 | 0 | 0 | 0 | 2 |
| [interpreter](interpreter.md) | 0 | 0 | 4 | 0 | 0 | 0 | 4 |
| [jit](jit.md) | 1 | 5 | 2 | 0 | 1 | 0 | 9 |
| [knobs](knobs.md) | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| [llvm](llvm.md) | 0 | 0 | 0 | 2 | 0 | 1 | 3 |
| [native-helpers](native-helpers.md) | 0 | 0 | 0 | 1 | 4 | 0 | 5 |
| [nvidia-driver](nvidia-driver.md) | 0 | 0 | 0 | 2 | 0 | 0 | 2 |
| [runtime-driver](runtime-driver.md) | 1 | 1 | 2 | 0 | 0 | 0 | 4 |
| [specialize](specialize.md) | 4 | 0 | 0 | 0 | 0 | 0 | 4 |
| **Total** | **8** | **13** | **11** | **7** | **5** | **1** | **45** |

## Component Files

- [autotuner](autotuner.md)
- [compiler](compiler.md) and [compiler-codegen](compiler-codegen.md)
- [experimental](experimental.md)
- [interpreter](interpreter.md)
- [jit](jit.md) and [async-compile](async-compile.md)
- [knobs](knobs.md)
- [llvm](llvm.md), [ir](ir.md), [src-main](src-main.md), and [src-passes](src-passes.md)
- [native-helpers](native-helpers.md)
- [nvidia-driver](nvidia-driver.md)
- [runtime-build](runtime-build.md) and [runtime-driver](runtime-driver.md)
- [specialize](specialize.md)
- [backends](backends.md), [cache](cache.md), [language](language.md), and [tools](tools.md)

## Auditing Guidance

When auditing, assign the issue a single rank:

- Use **Critical Blocker** if the issue affects the current free-threading goal
  and can plausibly crash, corrupt native/Python containers, execute the wrong
  kernel, or corrupt runtime/compiler state.
- Use **Blocker** if the issue affects the current free-threading goal without
  requiring debug-only modes or concurrent configuration mutation, but its
  expected consequence is less catastrophic than Critical Blocker.
- Use **Deferred** if the race is real but requires unsupported mid-flight
  configuration changes, hook mutation, driver/backend switching, or
  `TRITON_INTERPRET=1`.
- Use **Low** for latent, diagnostic, duplicate-work, or otherwise low-impact
  findings.
- Use **Rejected** or **Out-of-scope** when the report should preserve the
  analysis but should not ask Triton maintainers for a free-threading fix.
