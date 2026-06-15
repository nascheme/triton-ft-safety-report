# runtime/build.py and runtime/_allocation.py Free-Threading Issues

Scope: native extension build/load helpers in `runtime/build.py` and allocator
selection helpers in `runtime/_allocation.py`.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| 1 | Deferred | `_profile_allocator` | `has_profile_allocator()` / `_profile_allocator.get()` TOCTOU in launch hot path |
| 2 | Low | `_profile_allocator` | `_AllocatorWrapper._allocator` unsynchronized read/write |
| 3 | Low | `_compile_so` | Duplicate compile / `cache.put` write race for the same cache key |
| 4 | Low | `_compile_so` | Non-atomic multi-read of `knobs.build.*` during `_build` |
| 5 | Low | `_find_compiler` | `@functools.lru_cache` freezes first-seen `CC` / `CXX` / `PATH` resolution |
| 6 | Low | `_load_module_from_path` | Concurrent `exec_module` of the same `.so` cache file |

## Notes

- No Critical Blocker surfaced in this shallow pass.
- Issue #1 is Deferred because it requires profile allocator mutation during
  launch.
- Duplicate `_compile_so` work is mostly a build/cache efficiency concern; the
  resulting file path should converge.
- Re-check #6 if extension modules are imported through a path that does not
  already serialize module initialization.
