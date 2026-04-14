# `self.restore_copies` pre/post hook race

- **Status:** Open
- **Patch:** `cache-toctou.patch` (default pre/post hooks keep their
  snapshots in `self._tls` — a `threading.local` — so each thread has
  its own `restore_copies`)
- **Severity:** Significant
- **Component:** `python/triton/runtime/autotuner.py`

- **Shared state:** `self.restore_copies` — a `dict` on the shared `Autotuner`
  instance mapping tensor names to cloned snapshots. Written by the default
  `_pre_hook` (line 63), read and cleared by the default `_post_hook`
  (lines 73-75).
- **Writer(s):** `_pre_hook` at line 63: `self.restore_copies = {name: kwargs[name].clone() ...}`.
  `_post_hook` at line 75: `self.restore_copies = {}`.
- **Reader(s):** `_post_hook` at line 74: `kwargs[name].copy_(self.restore_copies[name])`.
- **Race scenario:** During benchmarking, `kernel_call()` runs `_pre_hook`
  then the kernel then `_post_hook` for each config trial. Thread A's
  `_pre_hook` saves snapshots of A's tensors. Thread B's `_pre_hook`
  overwrites `self.restore_copies` with B's tensor snapshots. Thread A's
  `_post_hook` then restores B's data into A's tensors (silent corruption),
  or Thread B's `_post_hook` clears the dict before Thread A reads it
  (`KeyError`). Shape/dtype mismatches between A's and B's tensors cause
  PyTorch `RuntimeError` on `copy_()`.
- **Consequences:** Silent wrong tensor contents after benchmark trials,
  missed restores, `KeyError`, or `RuntimeError` from incompatible
  cross-thread `copy_()`.
- **Suggested fix:** Make `restore_copies` a local variable captured by
  per-call hook closures rather than storing it on `self`.
