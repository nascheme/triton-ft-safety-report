# compiler/ Free-Threading Issues

Scope: `python/triton/compiler/`, especially `compiler.py` and the front-end
lowering path through `code_generator.py`.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| FT010 | HIGH | CompiledKernel | 2 | [`CompiledKernel._init_handles` lazy handle initialization race](compiler/compiled-kernel-init-handles-race.md) |
| FT008 | MED | compile() | 3 | [`knobs.runtime.add_stages_inspection_hook` TOCTOU in `compile()`](compiler/add-stages-inspection-hook-toctou.md) |
| FT011 | MED | HookChain | 3 | [`kernel_load_start_hook` / `kernel_load_end_hook` - `HookChain` iteration race in `_init_handles`](compiler/kernel-load-hook-chain-race.md) |
| FT009 | MED | code_generator | 1 | [`CodeGenerator.__init__` iterates live `fn.__globals__` via `get_capture_scope`](compiler/code-generator-gscope-iteration-race.md) |

## Notes

- FT010 is the only Tier 2 blocker in this component. The key bug is publishing
  `_run` before `module` / `function` are initialized.
- FT009 is Tier 1 because a non-Triton thread can mutate module globals while
  the Triton thread compiles.
- FT008 and FT011 are Tier 3 configuration/hook mutation during execution.
- `max_shared_mem`'s `lru_cache`, `AsmDict.__missing__`, filesystem cache
  coordination, backend selection reads, and `LazyDict` were reviewed and not
  promoted. They are either duplicate work, per-call state, or covered by the
  linked issues above.
