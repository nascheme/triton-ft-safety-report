# `_RUNTIME_BOOTSTRAP_CACHE` read-modify-write across `rendezvous()` calls

- **Status:** Open
- **Severity:** Significant
- **Component:** `experimental/gsan/symmetric_memory.py` (`rendezvous`)
- **Tier:** 2
- **Patch:** [`symmetric-memory-rendezvous-toctou.patch`](symmetric-memory-rendezvous-toctou.patch)

- **Shared state:** module-level `_RUNTIME_BOOTSTRAP_CACHE: dict[(id(process_group), device_index), set[int]]`. The value `set` records which peer ranks have already received this rank's runtime-state FD, so that subsequent `rendezvous()` calls only send the runtime-state FD to new peers.
- **Writer(s):** `rendezvous()` materializes the set via `setdefault(...)`, then later does `runtime_bootstrapped_peers.update(peers_needing_runtime_bootstrap)` once the protocol succeeds.
- **Reader(s):** `rendezvous()` builds a comprehension over `peer not in runtime_bootstrapped_peers` and uses the result to decide whether to call `export_runtime_state_handle(...)` and to attach a runtime-state FD to each per-peer `_send_fds()`.

- **Race scenario:** Two threads call `rendezvous(tensor, group)` for the same process group + device concurrently. `setdefault` returns the same shared `set` object to both.
  1. Thread A computes `peers_needing_runtime_bootstrap = {1, 2, 3}` (set is empty).
  2. Thread B computes the same `{1, 2, 3}` before A has run its `.update(...)`.
  3. Both threads call `export_runtime_state_handle(...)` and include a runtime-state FD in `_send_fds()` to every peer.
  4. Each peer's `_recv_fds()` therefore observes the `_RENDEZVOUS_FLAG_RUNTIME_STATE_FD` bit twice from this rank and calls `import_runtime_state_handle()` twice for the same source device. Runtime-state handles are imported once per peer by design; a duplicate import either rejects (raising in the middle of the rendezvous protocol) or installs a second mapping for the same device pair, corrupting downstream stream-sync bookkeeping.

  The `dict.setdefault` and `set.update` / `in` operations are individually safe on free-threaded CPython, but the *decision* "this peer still needs a runtime-state FD" is computed from a stale read and acted on without serialization.

- **Suggested fix:** The same lock that fixes the `_RENDEZVOUS_CACHE` check-then-act ([symmetric-memory-rendezvous-toctou](symmetric-memory-rendezvous-toctou.md)) naturally covers this region as well -- the read of `runtime_bootstrapped_peers`, the rendezvous protocol, and the trailing `runtime_bootstrapped_peers.update(...)` must all run under a single lock keyed on `(id(process_group), device_index)` (or a coarser lock).
