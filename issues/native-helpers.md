# native-helpers/

Scope: lower-frequency pybind11 bindings in `python/src/linear_layout.cc` and
`python/src/gluon_ir.cc`.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT033 | Low | `gluon_ir.cc` | [`layoutToGluon` static `GluonLayouts` cache: first-call import latency, leaked refs](native-helpers/gluon-layouts-static.md) |

## Notes

- FT033 is a magic-static initialization note. C++ serializes the initialization;
  the remaining concern is minor first-call import/refcount behavior.
