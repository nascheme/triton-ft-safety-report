# experimental/

Scope: `python/triton/experimental/`.

`experimental/gluon/` mostly inherits the core `jit` concurrency story.
`experimental/gsan/` is runtime-reachable once enabled and has real shared
state in its symmetric-memory and stream-sync helpers.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| 1 | Deferred | `experimental/gsan/_stream_sync.py` | `_compile_without_gsan()` mutates global `knobs.compilation.instrumentation_mode` via `knobs.compilation.scope()` |
| FT013 | Blocker | `experimental/gsan/symmetric_memory.py` | [`rendezvous()` does check-then-act on `_RENDEZVOUS_CACHE`](experimental/symmetric-memory-rendezvous-toctou.md) |
| FT012 | Blocker | `experimental/gsan/symmetric_memory.py` | [`_RUNTIME_BOOTSTRAP_CACHE[key]` returns a shared `set` later read/updated across rendezvous calls](experimental/symmetric-memory-bootstrap-cache-rmw.md) |
| 4 | Low | `experimental/gsan/_allocator.py` | `@functools.lru_cache()` helpers lazily compile/load the GSan allocator |
| 5 | Low | `experimental/gsan/_stream_sync.py` | `_runtime_state_layout` and `_compiled_sync_kernel` `lru_cache()` on hot sync path |
| 6 | Low | `experimental/gsan/symmetric_memory.py` | `_get_mem_pool` `lru_cache()` on first `empty()` path |
| 7 | Low | `experimental/gsan/_utils.py` | `_DLPACK_STATE` dict written from Python and popped from a C deleter |

## Notes

- FT013 and FT012 are the actionable Blocker findings.
- Issue #1 is Deferred because it mutates global knob state during
  compile/launch.
- The `lru_cache()` items are low-risk duplicate work unless their initialization
  side effects become non-idempotent.
- Gluon-specific state did not produce separate findings in this pass.
