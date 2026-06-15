# experimental/

Scope: `python/triton/experimental/`.

`experimental/gluon/` mostly inherits the core `jit` concurrency story.
`experimental/gsan/` is runtime-reachable once enabled and has real shared
state in its symmetric-memory and stream-sync helpers.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | MED | `experimental/gsan/_stream_sync.py` | 3 | `_compile_without_gsan()` mutates global `knobs.compilation.instrumentation_mode` via `knobs.compilation.scope()` |
| FT013 | MED | `experimental/gsan/symmetric_memory.py` | 2 | [`rendezvous()` does check-then-act on `_RENDEZVOUS_CACHE`](experimental/symmetric-memory-rendezvous-toctou.md) |
| FT012 | MED | `experimental/gsan/symmetric_memory.py` | 2 | [`_RUNTIME_BOOTSTRAP_CACHE[key]` returns a shared `set` later read/updated across rendezvous calls](experimental/symmetric-memory-bootstrap-cache-rmw.md) |
| 4 | LOW | `experimental/gsan/_allocator.py` | 1 | `@functools.lru_cache()` helpers lazily compile/load the GSan allocator |
| 5 | LOW | `experimental/gsan/_stream_sync.py` | 1 | `_runtime_state_layout` and `_compiled_sync_kernel` `lru_cache()` on hot sync path |
| 6 | LOW | `experimental/gsan/symmetric_memory.py` | 1 | `_get_mem_pool` `lru_cache()` on first `empty()` path |
| 7 | LOW | `experimental/gsan/_utils.py` | 2 | `_DLPACK_STATE` dict written from Python and popped from a C deleter |

## Notes

- FT013 and FT012 are the actionable Tier 2 findings.
- Issue #1 is Tier 3 because it mutates global knob state during compile/launch.
- The `lru_cache()` items are low-risk duplicate work unless their initialization
  side effects become non-idempotent.
- Gluon-specific state did not produce separate findings in this pass.
