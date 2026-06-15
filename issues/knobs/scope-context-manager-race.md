# `base_knobs.scope()` corrupts shared knob state and `os.environ` under concurrent use

- **Issue-Id:** FT027
- **Status:** Open
- **Rank:** Deferred
- **Component:** `python/triton/knobs.py` (`base_knobs.scope`)

- **Shared state:** `self.__dict__` of the module-level knob singleton on
  which `scope()` is invoked (e.g. `knobs.compilation`), and `os.environ`
  for every key in `self.knob_descriptors`.

- **Writer(s):**
  - `scope()` on `__exit__`: `self.__dict__.clear()` then
    `self.__dict__.update(orig)`, then per-key `os.environ[k] = v` /
    `del os.environ[k]`.
  - `env_base.__set__` on any explicit `knobs.<group>.<name> = ...`
    assignment, which writes `obj.__dict__[self.name]` and
    `os.environ[self.key]`.
  - `base_knobs.reset()` via `delattr`.
  - The body of the `with` block, which typically assigns overrides such
    as `triton.knobs.compilation.instrumentation_mode = ""`
    (`experimental/gsan/_stream_sync.py`).

- **Reader(s):** every `env_base.__get__` on this singleton, hit on every
  compile/launch path that consults a knob; plus native code reading the
  same env vars directly via `getenv` (e.g. `libtriton` and backend
  shared objects that consult `TRITON_*` at runtime).

- **Race scenario:**

  1. *Concurrent `scope()` on the same singleton.* Thread A enters
     `knobs.compilation.scope()` and snapshots
     `orig_A = dict(self.__dict__)`. Thread B enters
     `knobs.compilation.scope()` before A has assigned any override and
     snapshots `orig_B == orig_A`. Both threads then assign overrides
     inside their `with` blocks. When Thread A exits first, it runs
     `self.__dict__.clear()` then `self.__dict__.update(orig_A)`,
     erasing every override Thread B is currently relying on inside its
     own still-active `with` block.

  2. *Reader observes the `clear() / update()` window.* On exit, between
     `self.__dict__.clear()` and `self.__dict__.update(orig)`, a
     concurrent reader sees every knob fall through to its `env_base`
     default — even knobs the scope never touched.

  3. *`scope()` vs unrelated concurrent writers.* A writer doing
     `knobs.compilation.dump_ir = True` while a `scope()` is active has
     that override silently clobbered when the scope exits, regardless
     of whether `scope()` ever touched `dump_ir`.

  4. *`os.environ` restore TOCTOU.* The restore loop uses the same
     check-then-delete pattern as `setenv` (issue #6) and can raise
     `KeyError` from `del os.environ[k]` under concurrent `setenv` /
     `scope()` exits.

- **Suggested fix:** `scope()` cannot be made thread-safe by adding a
  lock alone, because `__dict__.clear()` / `update(orig)` is fundamentally
  destructive of any concurrent override. Either document `scope()` as
  single-threaded only and have callers serialize, or rework overrides
  to live on a thread-local / `ContextVar` stack so each thread's scope
  is isolated, skipping the `os.environ` round-trip entirely.
