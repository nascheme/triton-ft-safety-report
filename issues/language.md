# language/

Scope: `python/triton/language/`.

This package is mostly declarative DSL surface. Most functions are used during
compilation and inherit the concurrency story of `jit`, `compiler`, and
`specialize`.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | LOW | `language/core.py` | 2 | `tuple_type.name` uses `@cached_property`; duplicate first-use work if an instance is shared |
| 2 | LOW | `language/core.py` | 1 | `dtype` singletons should remain read-only after import |
| 3 | LOW | `language/core.py` | 1 | `_tensor_member_fn` and similar decorators mutate classes at import time |
| 4 | LOW | `language/core.py` | 1 | `_aggregate()` dynamically builds classes at decorator application time |
| 5 | LOW | `language/target_info.py` | 2 | `current_target()` reads `driver.active`; covered by `runtime-driver` |
| 6 | LOW | `language/extra/__init__.py` | 1 | Module discovery mutates `_backends` / `sys.modules` under the import lock |

## Notes

- No MED or HIGH issues were found.
- `semantic.py` constructs `TritonSemantic` per compile with a fresh builder.
- `math.py`, `random.py`, `standard.py`, and `extra/libdevice.py` are mostly
  `@jit` functions or stubs with no persistent mutable module state.
- Items #2, #3, and #6 are import-time write-once patterns unless a future path
  mutates them after import.
