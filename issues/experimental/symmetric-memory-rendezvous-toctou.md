# `rendezvous()` check-then-act on `_RENDEZVOUS_CACHE`

- **Status:** Open
- **Patch:** `symmetric-memory-rendezvous-toctou.patch` (module-level
  `threading.Lock` held across the `_RENDEZVOUS_CACHE` check, the
  rendezvous protocol, and the trailing cache write; the same lock
  covers [[symmetric-memory-bootstrap-cache-rmw]])
- **Severity:** Significant
- **Component:** `experimental/gsan/symmetric_memory.py` (`rendezvous`)
- **Tier:** 2

- **Shared state:** module-level `_RENDEZVOUS_CACHE: WeakValueDictionary[(base_ptr, storage_key, id(process_group)), GSanSymmetricMemoryHandle]`. The cache memoizes the full result of a multi-step Unix-socket FD-exchange rendezvous so that repeated `rendezvous()` calls on the same `(tensor storage, process group)` return the same handle.
- **Writer(s):** `rendezvous()` writes `_RENDEZVOUS_CACHE[cache_key] = handle` at the end of the protocol. `GSanSymmetricMemoryHandle.close()` (also reachable from `__del__`) removes the entry with `_RENDEZVOUS_CACHE.pop(self._cache_key)`.
- **Reader(s):** `rendezvous()` reads with `cached = _RENDEZVOUS_CACHE.get(cache_key)` near the top and returns early if `cached is not None and not cached._closed`.

- **Race scenario:** Two threads call `rendezvous(tensor, group)` on the same tensor/group concurrently (Tier 2: multiple Triton/gsan threads sharing a single process):
  1. Thread A: `cached = _RENDEZVOUS_CACHE.get(cache_key)` -> miss.
  2. Thread B: `cached = _RENDEZVOUS_CACHE.get(cache_key)` -> miss.
  3. Both run the full multi-step rendezvous (export FDs, `all_gather_object`, broadcast a `uuid` token, bind a Unix socket on a token-derived path, exchange FDs, `import_allocation_handles` for every peer, `import_runtime_state_handle`).
  4. Both finally do `_RENDEZVOUS_CACHE[cache_key] = handle`. The second write replaces the first; the first handle is no longer cached and is dropped on the next GC cycle.
  5. When the orphaned handle's `__del__` runs, `close()` calls `free_allocation(ptr, device_index)` on every imported peer pointer. Those same pointers are still live in the second (cached) handle -> use-after-free / double-free of peer allocations the second handle expects to use.

  Even without the close path, both threads have issued duplicate `import_allocation_handles` / `import_runtime_state_handle` for every peer, and both peer ranks have processed two concurrent FD-exchange sessions for the same logical key, distorting the distributed protocol's expected 1:1 connection count per peer per call.

- **Suggested fix:** Serialize the check-then-act region. Either guard the whole `rendezvous()` body with a module-level `threading.Lock`, or use a per-`cache_key` lock obtained from a small registry so only colliding keys serialize. The lock must cover the `_RENDEZVOUS_CACHE.get(...)` check, the protocol, and the final `_RENDEZVOUS_CACHE[cache_key] = handle` assignment.
