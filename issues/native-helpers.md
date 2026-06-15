# native-helpers/

Scope: lower-frequency pybind11 bindings in `python/src/linear_layout.cc` and
`python/src/gluon_ir.cc`.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| FT035 | - | `linear_layout.cc` | - | [Shared MLIRContext, linear_layout (rejected)](native-helpers/shared-mlir-context.md) |
| FT034 | - | `linear_layout.cc` | - | [`__imul__` mutates a shared `LinearLayout` Python object in place (rejected)](native-helpers/imul-shared-mutation.md) |
| FT031 | - | `linear_layout.cc` | - | [`get_shared_view` / `get_distributed_view` use `const_cast`; const-correctness only](native-helpers/const-cast-views.md) |
| FT032 | - | `gluon_ir.cc` | - | [Shared MLIRContext, gluon_ir (rejected)](native-helpers/gluon-builder-context.md) |
| FT033 | LOW | `gluon_ir.cc` | 1 | [`layoutToGluon` static `GluonLayouts` cache: first-call import latency, leaked refs](native-helpers/gluon-layouts-static.md) |

## Notes

- The original MLIRContext concern was rejected: construction-time
  `MLIRContext::Threading::DISABLED` does not disable StorageUniquer locks.
- The `LinearLayout` mutation concerns are public-API caller-contract issues,
  not current free-threading blockers.
- FT033 is a magic-static initialization note. C++ serializes the initialization;
  the remaining concern is minor first-call import/refcount behavior.
