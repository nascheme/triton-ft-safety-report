# `FunctionRewriter._compile_and_exec` mutates user kernel's `fn.__globals__`

- **Issue-Id:** FT015
- **Status:** Open
- **Severity:** MED
- **Component:** `runtime/interpreter.py` (`FunctionRewriter._compile_and_exec`)
- **Tier:** 2

- **Shared state:** `self.fn.__globals__` — the user kernel module's
  globals dict, shared across every thread that imports / uses that
  module.
- **Writer(s):** `_compile_and_exec` runs:

  ```python
  fn_globals = self.fn.__globals__
  for key, value in globals().items():
      if key not in fn_globals:
          fn_globals[key] = value
  exec(compiled_code, fn_globals, local_namespace)
  ```

  Triggered (per kernel) by `InterpretedFunction.rewrite()`, which is
  guarded only by a check-then-set on the class-level `rewritten_fn`
  dict (see issue #5 — duplicate rewrite is possible). Each invocation
  injects interpreter internals (`interpreter_builder`,
  `interpreter_semantic`, `_implicit_cvt`, `np`, etc.) into the user
  module.
- **Reader(s):** the kernel body executed by `exec`, any other code in
  the user's kernel module that reads its module globals, and any
  concurrent caller of `_patch_lang(fn)` which iterates
  `fn.__globals__.items()` looking for `tl` / `tl.core` (see issue #6).
- **Race scenario:** Thread A is in the `for key, value in
  globals().items():` copy loop, partway through populating
  `fn_globals`. Thread B starts an interpreted run of the same kernel
  and enters `_patch_lang(fn)`, which does
  `[value for _, value in fn.__globals__.items() if ...]`. B's
  iteration sees an undefined element sequence (per
  `PYTHON_THREADSAFETY.md`); B's `langs` list may miss `tl` or `tl.core`
  and silently fail to patch one of them, yielding wrong-kernel
  execution or an `InterpreterError`. Separately, the
  `if key not in fn_globals: fn_globals[key] = value` check-then-act
  pattern persistently leaks interpreter internals into the user's
  module namespace.
- **Suggested fix:** stop mutating `self.fn.__globals__`. Build a merged
  dict (`{**fn.__globals__, **globals()}`) once and pass it as the
  `globals` arg to `exec`. The user module's globals are then never
  observed mid-mutation, and interpreter internals do not leak into
  the user's namespace.
