# `CudaUtils` broken singleton pattern

- **Status:** Open
- **Patch:** `cudautils-singleton-race.patch` (replace broken singleton with double-checked locking under a class-level `threading.Lock`; move init body into `_init()` and make `__init__` a no-op)
- **Severity:** Minor
- **Component:** `third_party/nvidia/backend/driver.py`

- **Shared state:** `CudaUtils.instance` class attribute (line 62); all
  `self.*` instance attributes (lines 83–90); the five module globals
  from [issue #1](module-globals-lazy-init-race.md).
- **Writer(s):** `CudaUtils()` — `__new__` has a TOCTOU on
  `hasattr(cls, "instance")` (lines 59–61), and `__init__` re-runs on
  every call because `type.__call__` always invokes `__init__` on the
  returned object, even for a cached instance.
- **Reader(s):** `CudaLauncher.__init__` reads instance attributes like
  `utils.launch` and `utils.build_signature_metadata` (lines 300, 183).
- **Race scenario:** Two concurrent `CudaDriver()` calls both enter
  `CudaUtils()`.  Both run `compile_module_from_src` and `dlopen` the
  same `.so`.  The values produced are functionally equivalent (same
  deterministic inputs), so this is duplicate work, not data corruption.
- **Suggested fix:** Replace the broken singleton with a locked
  init-once pattern: do all work in a `_init()` method called under a
  class-level `threading.Lock` with double-checked locking, make
  `__init__` a no-op, and publish `cls._instance` only after `_init`
  completes.  See README.md triage notes for details.
