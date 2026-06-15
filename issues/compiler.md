# compiler/ Free-Threading Issues

Scope: `python/triton/compiler/`, especially `compiler.py` and the front-end
lowering path through `code_generator.py`.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT010 | Critical Blocker | CompiledKernel | [`CompiledKernel._init_handles` lazy handle initialization race](compiler/compiled-kernel-init-handles-race.md) |
| FT009 | Blocker | code_generator | [`CodeGenerator.__init__` iterates live `fn.__globals__` via `get_capture_scope`](compiler/code-generator-gscope-iteration-race.md) |
| FT008 | Deferred | compile() | `compile()` reads `knobs.runtime.add_stages_inspection_hook` more than once, so concurrent hook mutation can raise or mis-key cache state. Deferred because mid-flight hook mutation is outside the current support goal. |
| FT011 | Deferred | HookChain | `_init_handles` iterates `kernel_load_start_hook` / `kernel_load_end_hook` while concurrent `HookChain.add` / `remove` could mutate the chain. Deferred because hook mutation during kernel load is unsupported. |

## Notes

- FT010 is the only Critical Blocker in this component. The key bug is
  publishing `_run` before `module` / `function` are initialized.
- FT009 is a current-goal Blocker because a non-Triton thread can mutate module
  globals while the Triton thread compiles.  Patch is yet to be developed for this.
- FT008 and FT011 are Deferred configuration/hook mutation during execution.
- `max_shared_mem`'s `lru_cache`, `AsmDict.__missing__`, filesystem cache
  coordination, backend selection reads, and `LazyDict` were reviewed and not
  promoted. They are either duplicate work, per-call state, or covered by the
  linked issues above.
