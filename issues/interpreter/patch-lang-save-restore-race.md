# `_patch_lang` / `_LangPatchScope` save/restore permanently corrupts `tl`

- **Status:** Open
- **Severity:** SEVERE
- **Component:** `runtime/interpreter.py` (`_patch_lang`, `_LangPatchScope`)
- **Tier:** 2

- **Shared state:** attributes of the shared module/class objects `tl`,
  `tl.core`, `tl.tensor`, `tl.dtype`, `tl.math`, and
  `tl.core.tensor_descriptor_base`. Every interpreted kernel rewrites
  many of these (`tl.range`, `tl.static_assert`, `tl.reduce`,
  `tl.associative_scan`, `tl.tensor.__bool__`, `tl.tensor.__repr__`,
  `tl.tensor.T`, `tl.dtype.to_ir`, every `tl.core.is_builtin(...)`
  member, etc.) and tries to restore them in a `finally`.
- **Writer(s):** `_LangPatchScope.set_attr` does
  `original = getattr(obj, name, _MISSING); setattr(obj, name, value)`
  for every patched attribute. `_LangPatchScope.restore()` walks
  `_changes` in reverse and `setattr`s `original` back. Driven from
  `GridExecutor.__call__` via `patch_scope = _patch_lang(self.fn)`
  then `patch_scope.restore()` in `finally`.
- **Reader(s):** every consumer of `tl.*` process-wide — interpreter
  threads, non-interpreter threads that touch `tl.tensor` instances,
  and any other code in the process that imports `triton.language`.
- **Race scenario:** Thread A enters `_patch_lang`, reads original
  `tl.range -> R0`, sets `tl.range = R1_A`. Before A finishes, Thread B
  enters `_patch_lang`, reads "original" `tl.range -> R1_A` (A's patched
  value) and sets `tl.range = R1_B`. A finishes its kernel and restores
  `tl.range = R0`. B finishes and restores `tl.range = R1_A`. **`tl`
  module is now pinned to the interpreter-patched version for the rest
  of the process**, breaking every subsequent GPU-path kernel that uses
  `tl.range`. The same race corrupts `tl.tensor.__bool__`, `__repr__`,
  the `T` descriptor, etc., which leak into non-interpreter threads
  manipulating `tl.tensor` instances.
- **Suggested fix:** stop monkey-patching shared module/class objects.
  Route `tl.*` dispatch through a `ContextVar` / thread-local "semantic"
  selector. As a stopgap, serialize all interpreter calls with a
  process-wide lock and document the interpreter as single-threaded
  (matching the `README.md` "pragmatically out of scope" position).
