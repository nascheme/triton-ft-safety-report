# runtime/cache.py Free-Threading Issues

Scope: `python/triton/runtime/cache.py`.

The file mainly relies on POSIX atomic replace and `functools.lru_cache`
locking. The material cache races are caller-side and are covered in
[`compiler.md`](compiler.md), [`jit.md`](jit.md), and
[`autotuner.md`](autotuner.md).

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| 1 | Low | FileCacheManager.put | `os.removedirs(temp_dir)` can remove shared empty parent dirs during concurrent cleanup |
| 2 | Low | RemoteCacheManager / RedisRemoteCacheBackend | Non-atomic multi-read of `knobs.cache.redis.*` / `knobs.cache.remote_manager_class` during construction |
| 3 | Low | triton_key | `triton_key()` `@functools.lru_cache` cold-start / filesystem-dependent hashing |
| 4 | Low | FileCacheManager.get_file | `get_file` returns a path that can disappear before the caller opens it |

## Notes

- No Critical Blocker or Blocker issues were found in this file.
- If hardening #1, replace `os.removedirs(temp_dir)` with `os.rmdir(temp_dir)`;
  this function only owns the leaf temp directory.
- `triton_key()` duplicate cold-start work is acceptable; the cache decorator
  protects its internal dictionary.
- `get_file` is a caller contract issue: consumers must tolerate a cache file
  disappearing after lookup.
