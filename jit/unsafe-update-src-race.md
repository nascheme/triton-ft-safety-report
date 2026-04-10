# `JITCallable._unsafe_update_src` unsynchronized hash invalidation

- **Status:** Open
- **Severity:** Significant
- **Component:** `runtime/jit.py`

- **Shared state:** `JITCallable._src` and `JITCallable.hash` -- must stay in
  sync (`hash` is derived from `_src`). Every path except `_unsafe_update_src`
  uses `self._hash_lock` to enforce this.
- **Writer(s):** `_unsafe_update_src` (lines 543-551) writes `self.hash = None`
  then `self._src = new_src` **without taking `_hash_lock`**. The
  lock-respecting writer is `cache_key` (lines 499-520).
- **Reader(s):** `cache_key` reads both `self.hash` and `self._src` under
  `_hash_lock`
- **Race scenario:** Thread A holds `_hash_lock` inside `cache_key`, reads old
  `_src`, and is computing the hash. Thread B calls `_unsafe_update_src`,
  writes `self.hash = None` and `self._src = new_src` without the lock.
  Thread A finishes and writes `self.hash = <hash of old src>`. Thread B's
  invalidation (`hash = None`) is clobbered -- `self.hash` now points at the
  old-src hash while `self._src` holds `new_src`. Subsequent `cache_key`
  calls return the stale hash, causing compiles for the new source to reuse
  the old source's cache entry.
- **Suggested fix:** Acquire `self._hash_lock` inside `_unsafe_update_src` so
  the src replacement and hash invalidation cannot overlap with an in-flight
  `cache_key` computation.
