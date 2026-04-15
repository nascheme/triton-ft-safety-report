# `CompiledKernel._init_handles` lazy handle initialization race

- **Status:** Open
- **Patch:** `compiled-kernel-init-handles-race.patch` (per-instance `_init_lock` with double-check; publish handles before `self._run`; preserve poison-on-error)
- **Severity:** SEVERE
- **Component:** `python/triton/compiler/compiler.py`
- **Tier:** 2

- **Shared state:** `self._run`, `self.module`, `self.function`,
  `self.n_regs`, `self.n_spills`, `self.n_max_threads` on a
  `CompiledKernel` instance shared via
  `JITFunction.device_caches[device].kernel_cache`.
- **Writer(s):** `_init_handles()` — sets `self._run` at line 455,
  then `self.module`/`self.function`/etc. at line 468. Not atomic;
  no lock.
- **Reader(s):** `run` property (line 476) guards on
  `self._run is None`; `jit.py:761` reads `kernel.function`
  immediately after.
- **Race scenario:** Thread A enters `_init_handles`, sets
  `self._run = launcher_cls(...)` (line 455), is preempted. Thread B
  hits the `run` property, sees `self._run is not None`, skips init,
  returns the launcher. `jit.py:761` reads `kernel.function` (still
  `None`) and passes it to the driver launch call — null function
  handle. Secondary: the `self.module is None` guard at line 440 has
  a TOCTOU allowing duplicate `load_binary` and duplicate
  `kernel_load_*_hook` calls; the multi-target tuple-unpack at
  line 468 is not atomic across its five `STORE_ATTR` opcodes.
- **Suggested fix:** Add a per-instance `threading.Lock`. Hold it in
  `_init_handles`; double-check `self._run is not None` inside.
  Write `self._run` **last** on success (after module/function/etc.)
  so it serves as the publish flag. On error, poison `self._run`
  with a raising partial (preserving existing contract) before
  re-raising. See README.md triage notes for full fix code.
