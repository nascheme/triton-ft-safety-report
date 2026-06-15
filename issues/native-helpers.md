# native-helpers/

Scope: lower-frequency pybind11 bindings in `python/src/linear_layout.cc` and
`python/src/gluon_ir.cc`.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT033 | Low | `gluon_ir.cc` | `layoutToGluon` uses a function-local static `GluonLayouts` cache whose initialization is C++-serialized but may do first-call Python import/ref work. The remaining concern is minor cold-start behavior. |

## Notes

- FT033 is a magic-static initialization note. C++ serializes the initialization;
  the remaining concern is minor first-call import/refcount behavior.
