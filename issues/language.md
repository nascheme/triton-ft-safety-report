# language/

Scope: `python/triton/language/`.

This package is mostly declarative DSL surface. Most functions are used during
compilation and inherit the concurrency story of `jit`, `compiler`, and
`specialize`.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| 1 | Low | `language/core.py` | `tuple_type.name` uses `@cached_property`; duplicate first-use work if an instance is shared |
| 2 | Low | `language/core.py` | `dtype` singletons should remain read-only after import |
| 3 | Low | `language/core.py` | `_tensor_member_fn` and similar decorators mutate classes at import time |
| 4 | Low | `language/core.py` | `_aggregate()` dynamically builds classes at decorator application time |
| 5 | Low | `language/target_info.py` | `current_target()` reads `driver.active`; covered by `runtime-driver` |
| 6 | Low | `language/extra/__init__.py` | Module discovery mutates `_backends` / `sys.modules` under the import lock |

## Notes

- No Critical Blocker or Blocker issues were found.
- `semantic.py` constructs `TritonSemantic` per compile with a fresh builder.
- `math.py`, `random.py`, `standard.py`, and `extra/libdevice.py` are mostly
  `@jit` functions or stubs with no persistent mutable module state.
- Items #2, #3, and #6 are import-time write-once patterns unless a future path
  mutates them after import.
