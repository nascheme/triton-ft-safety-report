# `JITFunction.device_caches` defaultdict auto-vivification race

- **Status:** Open
- **Severity:** SEVERE
- **Component:** `runtime/jit.py`

- **Shared state:** `self.device_caches` -- a `defaultdict(self.create_binder)`
  on a shared `JITFunction` instance (line 790)
- **Writer(s):** `defaultdict.__missing__` on first access for each device key,
  triggered by `run()` (720), `preload()` (816), `_do_compile()` (860)
- **Reader(s):** Same call sites -- the returned tuple is immediately unpacked
  and used for the rest of the call
- **Race scenario:** Thread A and B both call `run()` with `device=0` for the
  first time. Both trigger `__missing__`, both invoke `create_binder`, both
  get separate tuples containing fresh empty `kernel_cache` /
  `kernel_key_cache` dicts. The second insertion overwrites the first. The
  losing thread holds orphaned caches -- any kernels it compiles are silently
  lost. Subsequent calls on the surviving entry recompile those kernels.
- **Suggested fix:** Replace the `defaultdict` with a plain dict and explicit
  double-checked locking under a per-instance lock. All call sites use a
  `_get_device_cache(device)` method instead of `self.device_caches[device]`.
