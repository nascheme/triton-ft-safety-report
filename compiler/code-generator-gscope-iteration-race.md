---
name: CodeGenerator gscope iteration over live __globals__
description: CodeGenerator.__init__ iterates fn.__globals__ directly during compile; concurrent module-level assignment produces undefined iteration sequence, causing spurious NameError or wrong globals captured into gscope
type: project
---

# Free-Threading Audit: `compiler/code_generator.py`

## Architecture Summary

`code_generator.py` lowers a parsed Triton `@jit` function body into TTIR.
`ast_to_ttir` (line 1620) constructs a fresh `CodeGenerator` per compile call
and drives `generator.visit(fn.parse())`. The generator's `__init__` seeds
`self.gscope` — the snapshot of module-level names visible to the kernel —
from whatever `JITFunction.get_capture_scope()` returns.

`JITFunction.get_capture_scope()` (`runtime/jit.py:492-497`):

```python
def get_capture_scope(self):
    fn = self.fn
    if fn.__closure__ is None:
        return self.__globals__
    nonlocals = {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
    return self.__globals__ | nonlocals
```

In the common no-closure case this returns **`fn.__globals__` itself** — the
live module dict of the module that defines the kernel — not a copy.

`CodeGenerator.__init__` then iterates it at `code_generator.py:308-318`:

```python
self.gscope = {}
for k, v in gscope.items():
    if isinstance(v, ModuleType):
        self.gscope[k] = module_map.get(v.__name__, v)
        continue
    module_name = getattr(v, "__module__", "")
    if module_name in module_map:
        self.gscope[k] = getattr(module_map[module_name], v.__name__)
    else:
        self.gscope[k] = v
```

The same construction path runs again for every @jit'd callee reached while
lowering the kernel (`code_generator.py:1332`), so any kernel that calls
sub-kernels re-runs this iteration for each callee's `__globals__` as well.

## Significant Issues

### `gscope.items()` iteration over live `fn.__globals__`

- **Shared state:** the defining module's `__dict__` (reached as
  `fn.__globals__`), which is a genuine process-global mutable `dict`.
- **Writer(s):** any Python thread executing code in the kernel's defining
  module that does module-level assignment — e.g. `module.some_flag = True`,
  `module.CONST = new_value`, lazy imports (`from x import y` at function
  scope ultimately rebinds into the importing module), or even a background
  thread updating a progress counter stored at module level. These are
  ordinary Python operations that were previously serialized by the GIL.
- **Reader(s):** `CodeGenerator.__init__` at `code_generator.py:309` during
  `ast_to_ttir`, i.e. on the compile hot path from `JITFunction._do_compile`
  and from `triton.compiler.compile`. Also repeated per callee at
  `code_generator.py:1332`.
- **Race scenario:** Thread A is compiling a kernel and enters
  `CodeGenerator.__init__`; `gscope` is `fn.__globals__` by identity (no
  closure). Thread B is executing code in the same Python module that
  assigns or deletes a module-level name. Per `PYTHON_THREADSAFETY.md` and
  the free-threading docs, single dict ops are atomic but **iteration
  under concurrent mutation has an undefined observed element sequence** —
  elements may be visited twice, skipped entirely, or observed in
  inconsistent order. Concrete consequences:

  1. A legitimate global the kernel references is skipped in the copy.
     Later `global_lookup` at `code_generator.py:373` misses it and raises
     `NameError: '<name> is not defined'` — the compile fails spuriously
     in a way that is not reproducible on re-run.
  2. A `constexpr` global is visited *after* thread B rebinds it. The
     kernel is lowered with the new value, but Triton's cache key (built
     earlier from `JITFunction.used_global_vals` in the jit.py hash
     path) was computed against the old value. The compiled kernel is
     then stored under a stale key — wrong-kernel reuse on subsequent
     launches with the old source.
  3. A module-level name that is temporarily removed (`del module.x`)
     during the iteration causes either a missed key or, for names that
     the kernel syntactically references, a `NameError` at compile time
     even though the name is perfectly valid before and after the race.

  All three are Tier 1 (compile-while-unrelated-Python-work) and also
  compound on the Tier 2 hot path when two threads first-compile
  distinct kernels defined in the same module.

- **Why the closure branch is safer (but not immune):** when
  `fn.__closure__` is not `None`, `get_capture_scope` returns
  `self.__globals__ | nonlocals`, which constructs a fresh merged dict
  that is call-local. The iteration at line 309 is then over a local
  dict and is safe. However, building the merged dict itself iterates
  `self.__globals__` inside `dict.__or__` — that path relies on CPython's
  per-dict critical section to produce a coherent copy. The no-closure
  branch bypasses the merge entirely and hands the live dict straight
  to Python-level iteration, so it is the primary hazard.

- **Why this is not redundant with the jit.py audit:** none of the
  existing jit writeups cover `get_capture_scope` or the
  `CodeGenerator` iteration. `cache_key` in `runtime/jit.py:509` also
  hands `self.__globals__` to `DependenciesFinder`, which walks it
  under `_hash_lock`; that is a separate iteration site with the same
  root cause and should be handled by the same fix.

- **Suggested fix:** make `get_capture_scope` always return a fresh
  dict, even on the no-closure path:

  ```python
  def get_capture_scope(self):
      fn = self.fn
      if fn.__closure__ is None:
          return dict(self.__globals__)
      nonlocals = {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
      return self.__globals__ | nonlocals
  ```

  `dict(d)` goes through `PyDict_Copy`, which under free-threading
  acquires the source dict's critical section and produces a coherent
  snapshot. Downstream iteration in `CodeGenerator.__init__` then runs
  over a call-local dict and cannot race with concurrent module-level
  assignment. The fix is a one-line change in `runtime/jit.py:492-497`
  and covers both the `ast_to_ttir` top-level call and the per-callee
  path at `code_generator.py:1332`. The `DependenciesFinder` iteration
  in `cache_key` should get the same treatment independently.

**Tier:** Primarily Tier 1 (compile concurrent with unrelated Python
module-level writes). Compounds on Tier 2 when multiple kernels from
the same module are first-compiled concurrently.

**Severity:** Significant. Not a crash — CPython's free-threaded dict
iteration does not corrupt memory — but the consequences range from
spurious `NameError` compile failures that cannot be reproduced to
stale cache-key / new-value mismatches that bake an inconsistent
constexpr into a reusable compiled kernel.

## Notes on other module-level state in this file

- `_condition_types = {bool, int, type(None)}` at line 108: written
  once at module import, never mutated. Read-only. Not a concern.
- `CodeGenerator.builtin_namespace` at lines 342-350: class-level
  dict populated at class-definition time by a comprehension plus
  `.update(...)`. Never mutated after class body execution. Read-only
  concurrent access (`in` / `.get` / `.values()`) is safe. Not a
  concern.
- `module_map` is read-only inside `code_generator.py` (only
  `module_map.get(...)`, `module_map[name]` lookups; no writes).
  Whether the caller passes a shared instance is a `compile()`-level
  concern, and `compile()` itself obtains `module_map` from the
  backend per call.
- `ast_to_ttir` constructs a fresh `CodeGenerator` per call; there
  is no cross-call `CodeGenerator` instance state.
