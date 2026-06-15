# knobs.py Free-Threading Issues

Scope: `python/triton/knobs.py`, including process-global configuration
singletons, hook chains, callback slots, and `os.environ` side effects.

All issues here are Deferred: concurrent configuration mutation while other
threads compile or launch is tracked but not a current free-threading support
goal.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| 1 | Deferred | HookChain | `HookChain.__call__` iterates `self.calls` under concurrent `add`/`remove` |
| 2 | Deferred | HookChain | `HookChain.add` / `HookChain.remove` TOCTOU on membership check |
| 3 | Deferred | runtime hook slots | Hot-path callers re-read `knobs.runtime.*_hook` across `is not None` / call |
| FT027 | Deferred | base_knobs.scope | `base_knobs.scope()` saves and restores shared descriptor state plus `os.environ`, so overlapping scopes can restore the wrong values. Deferred because concurrent global configuration mutation is unsupported. |
| 5 | Deferred | env_base.__set__ | `env_base.__set__` non-atomic instance-dict vs `os.environ` update |
| 6 | Low | setenv | `setenv` check-then-delete TOCTOU on `os.environ` |
| 7 | Low | base_knobs.reset | `base_knobs.reset()` non-atomic multi-descriptor delete |
| 8 | Low | refresh_knobs | `refresh_knobs()` vs concurrent readers on `runtime.debug` / `compilation.instrumentation_mode` |

## Notes

- `HookChain` copy-on-write would address #1 and #2 and also help compiler/JIT
  hook users.
- Hot-path double-read call sites are tracked where they occur, including the
  `add_stages_inspection_hook` row in `jit.md`.
- `base_knobs.scope()` is already documented as unsafe across threads; FT027
  records the concrete failure mode.
- Atomicity of individual dict/attribute operations is not enough here because
  the bugs are multi-step configuration transactions.
