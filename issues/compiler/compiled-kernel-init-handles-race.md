# `CompiledKernel._init_handles` lazy handle initialization race

- **Status:** Open
- **Severity:** HIGH
- **Component:** `python/triton/compiler/compiler.py`
- **Tier:** 2
- **Patch:** [`compiled-kernel-init-handles-race.patch`](compiled-kernel-init-handles-race.patch)

- **Shared state:** `self._run`, `self.module`, `self.function`,
  `self.n_regs`, `self.n_spills`, `self.n_max_threads` on a
  `CompiledKernel` instance shared via
  `JITFunction.device_caches[device].kernel_cache`.
- **Writer(s):** `_init_handles()` — sets `self._run` first, then
  `self.module`/`self.function`/etc. from the `load_binary` tuple-unpack.
  Not atomic; no lock.
- **Reader(s):** `run` property guards on `self._run is None`;
  `jit.py` (`JITFunction.run`) reads `kernel.function` immediately
  after.
- **Race scenario:** Thread A enters `_init_handles`, sets
  `self._run = launcher_cls(...)`, is preempted. Thread B hits the
  `run` property, sees `self._run is not None`, skips init, returns
  the launcher. The caller in `jit.py` reads `kernel.function` (still
  `None`) and passes it to the driver launch call — null function
  handle. Secondary: the `self.module is None` guard has a TOCTOU
  allowing duplicate `load_binary` and duplicate `kernel_load_*_hook`
  calls; the multi-target tuple-unpack is not atomic across its five
  `STORE_ATTR` opcodes.
- **Suggested fix:** Add a per-instance `threading.Lock`. Hold it in
  `_init_handles`; double-check `self._run is not None` inside.
  Write `self._run` **last** on success (after module/function/etc.)
  so it serves as the publish flag. On error, poison `self._run`
  with a raising partial (preserving existing contract) before
  re-raising. See issues.md triage notes for full fix code.
