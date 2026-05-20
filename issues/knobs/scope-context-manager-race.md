# `base_knobs.scope()` corrupts shared knob state and `os.environ` under concurrent use

- **Status:** Open
- **Severity:** Significant
- **Component:** `python/triton/knobs.py` (`base_knobs.scope`, lines 295-309)

- **Shared state:**
  - `self.__dict__` of the module-level knob singleton on which `scope()`
    is invoked (e.g. `knobs.compilation` at knobs.py:565). The singleton
    is a process-global object referenced from every compile/launch path.
  - `os.environ` for every key in `self.knob_descriptors`.

- **Writer(s):**
  - `scope()` itself, on `__exit__`: `self.__dict__.clear()` followed by
    `self.__dict__.update(orig)` (knobs.py:302-303), then per-key
    `os.environ[k] = v` / `del os.environ[k]` (knobs.py:305-309).
  - `env_base.__set__` (knobs.py:84-90) on any explicit
    `knobs.<group>.<name> = ...` assignment, which writes
    `obj.__dict__[self.name]` and `os.environ[self.key]`.
  - `base_knobs.reset()` (knobs.py:290-293) via `delattr`.
  - The body of the `with` block, which typically assigns overrides
    such as `triton.knobs.compilation.instrumentation_mode = ""`
    (`experimental/gsan/_stream_sync.py:39-41`).

- **Reader(s):**
  - Every `env_base.__get__` (knobs.py:75-79) on this singleton, hit on
    every compile/launch path that consults a knob (e.g. `dump_ir`,
    `instrumentation_mode`, `always_compile`).
  - Native code reading the same env vars directly via `getenv` — for
    example `from triton._C.libtriton import getenv` at knobs.py:15 and
    backend shared objects that consult `TRITON_*` at runtime.

- **Race scenario:**

  1. *Concurrent `scope()` on the same singleton.* Thread A enters
     `knobs.compilation.scope()` and snapshots
     `orig_A = dict(self.__dict__)`. Thread B enters
     `knobs.compilation.scope()` before A has assigned any override and
     snapshots `orig_B == orig_A`. Both threads then assign overrides
     inside their `with` blocks. When Thread A exits first, it runs
     `self.__dict__.clear()` followed by `self.__dict__.update(orig_A)`,
     erasing every override Thread B is currently relying on inside its
     own still-active `with` block. Any subsequent compile on Thread B
     consults the wrong knob value (e.g. `instrumentation_mode` reverts
     from `""` back to `"gsan"`, defeating the whole point of
     `_compile_without_gsan` in `experimental/gsan/_stream_sync.py:37-41`).

  2. *Reader observes the `clear() / update()` window.* On exit, between
     `self.__dict__.clear()` (knobs.py:302) and
     `self.__dict__.update(orig)` (knobs.py:303), a concurrent reader
     on another thread sees every knob fall through to its `env_base`
     default — even knobs the scope never touched. A compile dispatched
     in that window observes a process-wide reset.

  3. *`scope()` vs unrelated concurrent writers.* A writer on another
     thread doing `knobs.compilation.dump_ir = True` while a `scope()`
     is active has that override silently clobbered when the scope
     exits via `__dict__.clear(); __dict__.update(orig)`, regardless of
     whether `scope()` ever touched `dump_ir`.

  4. *`os.environ` restore TOCTOU.* The restore loop (knobs.py:305-309)
     uses the same check-then-delete pattern as `setenv` (issue #6); it
     races with concurrent `setenv` calls and concurrent `scope()`
     exits and can raise `KeyError` from `del os.environ[k]`.

- **Suggested fix:** `scope()` cannot be made thread-safe by adding a
  lock alone, because `__dict__.clear()` / `update(orig)` is fundamentally
  destructive of any concurrent override. Either:
  - document `scope()` as single-threaded only (it already mutates the
    process-global `os.environ`, which is inherently shared), and have
    callers serialize, or
  - rework overrides to live on a thread-local / `ContextVar` stack so
    each thread's scope is isolated from other threads' state, and skip
    the `os.environ` round-trip entirely under free-threading.
