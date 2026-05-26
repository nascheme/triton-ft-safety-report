# `InterpretedFunction.__call__` leaks `_patch_lang` patches

- **Status:** Open
- **Severity:** MED
- **Component:** `runtime/interpreter.py` (`InterpretedFunction.__call__`)
- **Tier:** 2

- **Shared state:** the same shared `tl.*` module/class attributes
  patched by `_patch_lang` (see `patch-lang-save-restore-race.md`).
- **Writer(s):** `InterpretedFunction.__call__` (the device-function
  call path) does:

  ```python
  def __call__(self, *args, **kwargs):
      _patch_lang(self.fn)
      fn = self.rewrite()
      try:
          return fn(*args, **kwargs)
      except Exception as e:
          raise InterpreterError(repr(e)) from e
  ```

  It throws away the `_LangPatchScope` returned by `_patch_lang`, so
  the patches it installs are **never restored**. Compare
  `GridExecutor.__call__`, which stores `patch_scope = _patch_lang(...)`
  and calls `patch_scope.restore()` in a `finally`.
- **Reader(s):** every subsequent reader of `tl.*` in the process —
  other interpreted runs, GPU-path threads that touch `tl.tensor`
  instances, the user's own code.
- **Race scenario:**
  - Under the GIL: each call to a device-helper `@triton.jit` function
    permanently overwrites `tl.range`, `tl.tensor.__bool__`, etc., with
    the interpreter wrappers. The outer `GridExecutor.__call__`'s
    `restore()` only reverts entries that *its* `_patch_lang` captured,
    not entries installed by nested `__call__` invocations.
  - Under free threading: combined with the save/restore race
    (`patch-lang-save-restore-race.md`), Thread A's nested `__call__`
    captures Thread B's live patches as "originals" and never restores
    them anyway — the two issues compound.
- **Suggested fix:** if `__call__` is only reachable from inside an
  active `GridExecutor` run (the outer scope already installed the
  patches), drop the inner `_patch_lang(self.fn)` call entirely.
  Otherwise capture the scope and restore it in a `finally`, matching
  `GridExecutor.__call__`.
