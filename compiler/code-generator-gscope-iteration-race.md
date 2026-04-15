# `CodeGenerator.__init__` iterates live `fn.__globals__`

- **Status:** Open
- **Patch:** `code-generator-gscope-iteration-race.patch` (snapshot `fn.__globals__` via `dict()` in `get_capture_scope` no-closure path)
- **Severity:** Significant
- **Component:** `python/triton/compiler/code_generator.py`
- **Tier:** 1

- **Shared state:** The defining module's `__dict__`, reached as
  `fn.__globals__` via `JITFunction.get_capture_scope()`
  (`runtime/jit.py:492–497`), which returns the live dict by identity
  in the common no-closure case.
- **Writer(s):** Any thread doing module-level assignment in the
  kernel's defining module.
- **Reader(s):** `CodeGenerator.__init__` iterates `gscope.items()`
  at `code_generator.py:309` during `ast_to_ttir` (compile hot path).
  Repeated per callee at line 1332.
- **Race scenario:** Thread A compiles a kernel; `gscope` is
  `fn.__globals__` by identity. Thread B assigns a module-level name
  in the same module. Iteration under concurrent mutation has undefined
  element sequence (per `PYTHON_THREADSAFETY.md`): a legitimate global
  may be skipped → spurious `NameError` at compile time, or a
  `constexpr` global is observed post-rebind while the cache key was
  computed against the old value → stale-keyed compiled kernel.
- **Suggested fix:** Make `get_capture_scope()` always return a
  snapshot:
  ```python
  if fn.__closure__ is None:
      return dict(self.__globals__)
  ```
  `dict()` goes through `PyDict_Copy`, which acquires the source
  dict's critical section under free-threading.
