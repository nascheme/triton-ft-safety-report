# `DriverConfig.default` lazy init race creates duplicate driver instances

- **Status:** Open
- **Patch:** `driver-default-lazy-init-race.patch` (add a `threading.Lock` to `DriverConfig` and use double-checked locking in `default`/`active`; also serializes `set_active`/`reset_active`. Also fixes `driver-active-set-active-race.md`.)
- **Severity:** Significant
- **Component:** `runtime/driver.py:30-34`
- **Tier:** 1

- **Shared state:** `DriverConfig._default` on the module-level singleton
  `driver`. Starts `None`, populated lazily by the `default` property.
- **Writer(s):** `default` getter at line 33
  (`self._default = _create_driver()`), no lock.
- **Reader(s):** `default` getter and transitively `active` getter
  (line 38-39), read from every hot path in the project.
- **Race scenario:** Thread A and B both call `driver.active` for the first
  time. Both read `self._default is None` → `True` and enter
  `_create_driver()` concurrently. For the NVIDIA backend this runs
  `compile_module_from_src` and `dlopen`s a native utils module twice in
  parallel. One driver instance is silently overwritten and leaked. The
  downstream `CudaUtils.__new__` singleton has its own layered TOCTOU.
  This is a plausible root cause of the segfault reported in
  [triton-lang/triton#6721](https://github.com/triton-lang/triton/issues/6721)
  (see [seg-fault-gh-6721.md](seg-fault-gh-6721.md)).
- **Suggested fix:** Serialize lazy init with a `threading.Lock` on
  `DriverConfig` using double-checked locking. The same lock should guard
  `active`, `set_active`, and `reset_active`
  (see [driver-active-set-active-race.md](driver-active-set-active-race.md)).
