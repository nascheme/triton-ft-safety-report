# python/src/specialize.cc Free-Threading Issues

Scope: native argument-specialization helpers behind `JITFunction`
specialization.

This is one of the highest-priority native files: it has process-global lazy
initialization guarded by a plain boolean and process-global
`std::unordered_map` caches on specialization hot paths.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT044 | Critical Blocker | `init_globals` / `init_called` | [`init_globals()` TOCTOU: plain-bool guard allows concurrent lazy init of all file-scope globals](specialize/init-globals-toctou.md) |
| FT042 | Critical Blocker | `dtype_ptr2str` / `dtype2str` | [`dtype_ptr2str` and `dtype2str` `std::unordered_map` caches mutated on the specialization hot path](specialize/dtype-ptr2str-unordered-map-race.md) |
| FT043 | Critical Blocker | `dtype2str` | [`dtype2str` `std::unordered_map` mutated on the tensordesc specialization path](specialize/dtype2str-unordered-map-race.md) |
| FT045 | Critical Blocker | `type_handler_cache` | [`type_handler_cache` read during / before its population races init and specialize-dispatch](specialize/type-handler-cache-init-race.md) |
| 5 | Blocker | `init_called` memory ordering | Non-atomic publish is subsumed by FT044 |
| 6 | Low | `torch_tensor_cls` lazy probe | `torch.Tensor` is discovered only if torch is already imported at first specialize call |

## Notes

- FT044 and FT045 are fixed together by serializing first-use initialization.
- FT042 and FT043 are sibling unordered-map cache races and share one patch.
- Issue #5 does not need a separate writeup once FT044 uses proper
  synchronization.
- Issue #6 is a performance/coverage note, not a correctness bug.
