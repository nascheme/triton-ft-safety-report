# runtime/build.py and runtime/_allocation.py Free-Threading Issues

Scope: native extension build/load helpers in `runtime/build.py` and allocator
selection helpers in `runtime/_allocation.py`.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | MED | `_profile_allocator` | 3 | `has_profile_allocator()` / `_profile_allocator.get()` TOCTOU in launch hot path |
| 2 | LOW | `_profile_allocator` | 3 | `_AllocatorWrapper._allocator` unsynchronized read/write |
| 3 | LOW | `_compile_so` | 2 | Duplicate compile / `cache.put` write race for the same cache key |
| 4 | LOW | `_compile_so` | 3 | Non-atomic multi-read of `knobs.build.*` during `_build` |
| 5 | LOW | `_find_compiler` | 1 | `@functools.lru_cache` freezes first-seen `CC` / `CXX` / `PATH` resolution |
| 6 | LOW | `_load_module_from_path` | 2 | Concurrent `exec_module` of the same `.so` cache file |

## Notes

- Nothing HIGH surfaced in this shallow pass.
- Issue #1 is the only MED item and is Tier 3 because it requires profile
  allocator mutation during launch.
- Duplicate `_compile_so` work is mostly a build/cache efficiency concern; the
  resulting file path should converge.
- Re-check #6 if extension modules are imported through a path that does not
  already serialize module initialization.
