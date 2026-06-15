# compiler-codegen/ Free-Threading Issues

Pass 4 deeper review of the compile-time front-end:

- `compiler/code_generator.py` — Python AST → MLIR IR translation, recursive
  `CodeGenerator` per-callee compilation, global-scope capture, builtin
  namespace, static lowering table.
- `compiler/make_launcher.py` — empty file (0 bytes); nothing to review.

The earlier `compiler/` pass (see [`compiler.md`](compiler.md))
already touched `code_generator.py` lightly and filed
[`code-generator-gscope-iteration-race.md`](compiler/code-generator-gscope-iteration-race.md).
This report is a wider, surface-level sweep intended to enumerate every
remaining candidate worth a deep-dive. Each entry below points at the question
another agent should answer before deciding whether to upgrade into a separate
issue file.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | MED | code_generator | 1 | [`gscope` iteration race in `CodeGenerator.__init__` and `call_JitFunction`](#1-gscope-iteration-race-already-filed) — already filed in `compiler/`; re-listed here for component completeness |
| 2 | Candidate | code_generator | 1/3 | [`module_map` value-module attribute reads](#2-module_map-target-module-attribute-reads-during-gscope-rewrite) |
| 3 | Candidate | code_generator | 1 | [`inspect.signature(fn)` caching on shared user callables](#3-inspectsignaturefn-on-user-callables-during-call-lowering) |
| 4 | Candidate | code_generator | 1 | [`warnings.warn` / `warn_explicit` filter state during compile](#4-warningswarn-warn_explicit-during-compile) |
| 5 | Candidate | code_generator | 2 | [`JITFunction._src` rebind during `parse()`](#5-jitfunction_src-rebind-during-parse) |
| 6 | Candidate | code_generator | 1/2 | [`self.builder.codegen_fns` / `module_map` / `options` attribute storage on `ir.builder`](#6-builder-attribute-storage-codegen_fns-module_map-options) |
| 7 | Candidate | code_generator | 1 | [Lazy intra-function imports of `TritonSemantic` / `GluonSemantic`](#7-lazy-intra-function-imports-of-the-semantic-classes) |
| 8 | Candidate | code_generator | 1 | [Backend-supplied `module_map` / `codegen_fns` identity across calls](#8-backend-supplied-module_map--codegen_fns-identity-across-compile-calls) |

No issue is upgraded out of "Candidate" status in this pass; deeper validation
is required before writing per-issue files. Severity column reflects the
**potential** ceiling under the worst plausible scenario, not a confirmed
finding.

## Triage notes

### Module-level and class-level state — confirmed safe

A grep for module-level mutable state on `code_generator.py`:

- `_condition_types = {bool, int, type(None)}` (module-level) — set
  literal of three immutable types; never mutated.
- `CodeGenerator.builtin_namespace` — class-body
  dict populated at class-definition time from `(len, list, range, float, int,
  isinstance, getattr, hasattr)` plus three explicit overrides; never mutated
  after class body finishes executing.
- `CodeGenerator._method_name_for_bin_op`, `_method_name_for_bool_op`,
  `_method_name_for_comp_op`, `_method_name_for_unary_op` — all class-body
  literal dicts, frozen.
- `CodeGenerator.statically_implemented_functions` — class-body literal dict,
  frozen.
- The `from ..experimental.gluon import language as ttgl` is inside the class
  body and runs at class-definition time, protected by Python's import lock.

None of these are reportable; they are all "write-once at import / class body
execution" per CLAUDE.md.

### Per-CodeGenerator instance state — per-call, not cross-thread

`CodeGenerator.__init__` builds the instance fields fresh per call:

- `self.gscope`, `self.lscope`, `self.local_defs`, `self.return_vals`,
  `self.return_ips`, `self.scf_stack`, `self.function_ret_types`, `self.module`,
  `self.builder`, `self.semantic`.

`function_ret_types` deserves a one-line callout: it is threaded into recursive
`CodeGenerator(...)` constructions for callee compilation in
`call_JitFunction` (`function_types=self.function_ret_types`) so the same dict
is **shared across recursion within one `ast_to_ttir` call**. The
TOCTOU at `self.module.has_function(fn_name)` →
either `self.function_ret_types[fn_name] = callee_ret_type` or
`self.function_ret_types[fn_name]` is benign within one call
(single thread of recursion) but would be a real race if anyone ever shared a
pre-built `CodeGenerator` across threads. `ast_to_ttir` always constructs a
fresh top-level `CodeGenerator` per `compile()` call, so this is currently
safe; flagging it as a property to re-check if `CodeGenerator` ever becomes
reusable.

### 1. `gscope` iteration race (already filed)

Already documented in
[`compiler/code-generator-gscope-iteration-race.md`](compiler/code-generator-gscope-iteration-race.md).

The relevant call sites in `code_generator.py` are:

- `CodeGenerator.__init__` (top-level kernel, gscope-rewrite loop)
- `call_JitFunction` (recursive callee compile)
- `ContainsReturnChecker.__init__` and its `gscope[name]` reads — also takes
  `self.gscope` by reference and walks the AST while reading from it. If
  `gscope` is the live `fn.__globals__` (no closure), then concurrent
  module-level rebinds in another thread make even these reads "iteration
  over a concurrently-mutated mapping" because `ast.NodeVisitor.generic_visit`
  walks an unbounded set of names and reads each one — same fix (snapshot)
  covers it.

The existing fix (snapshot in `JITFunction.get_capture_scope` no-closure
branch) repairs all three call sites at once. No new issue file required.

### 2. `module_map` target-module attribute reads during `gscope` rewrite

In `CodeGenerator.__init__`:

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

`module_map` is supplied by the backend (`get_module_map()` in
`third_party/nvidia/backend/compiler.py` and
`third_party/amd/backend/compiler.py`). For NVIDIA it is
`{"triton.language.extra.libdevice": libdevice}`. The dict itself is a fresh
literal per `compile()` call, but **its values are live module objects**.

`getattr(module_map[module_name], v.__name__)` reads `module.__dict__[name]`,
i.e. the module's globals dict, by attribute. Triton itself does not write to
`libdevice` after import, but third-party callers that monkey-patch
`triton.language.extra.libdevice.*` from another thread would race the reader.

Same shape as issue #1 (live module-dict iteration), but a single
`getattr` rather than `.items()` iteration. Single name lookup on a `dict`
under free-threading is internally safe (per `PYTHON_THREADSAFETY.md` — single
ops on dicts use the per-object critical section), so the dict read itself
won't corrupt; the only consequence is "stale value if someone monkey-patched
the module mid-compile."

**Investigation question:** does any first-party Triton path mutate
`triton.language.extra.libdevice` or any other `module_map` value after import?
If no, downgrade to "not worth reporting." If a backend or extra package does,
file as Tier 1 LOW. Same question applies to the `visit_Attribute`
fixed-point walk.

### 3. `inspect.signature(fn)` on user callables during call lowering

In the call-lowering branch of `visit_Call`:

```python
sig = getattr(fn, "signature", None)
if isinstance(fn, ConstexprFunction):
    extra_kwargs["_semantic"] = self.semantic
else:
    if sig is None:
        sig = inspect.signature(fn)
    if '_semantic' in sig.parameters:
        extra_kwargs["_semantic"] = self.semantic
    if '_generator' in sig.parameters:
        extra_kwargs['_generator'] = self
```

`inspect.signature(fn)` walks `fn` to construct a `Signature` object. CPython
caches the result on the callable in some paths. If `fn` is a shared
`triton.language.core.*` builtin reused across threads, two threads invoking
`inspect.signature` on it concurrently could race on the cache fill.

CPython's `inspect.Signature.from_callable` is implemented in pure Python and
does mutate `fn.__signature__` in some branches. Under free-threading, that
mutation is a single attribute set (atomic) and the value is idempotent — both
threads compute the same Signature for the same callable. Likely benign.

**Investigation question:** does any callable reachable through this path
have a `signature` attribute that is a *mutable* object whose mutation is
load-bearing? Quick `grep` on `signature =` in `triton.language.core` would
answer this.

### 4. `warnings.warn` / `warn_explicit` during compile

Two warning emission sites:

- block-tensor `if` deprecation warning, fired when the user passes a block
  tensor to a non-`item()`-converted `if`.
- boolean `and`/`or` on non-scalar tensors deprecation warning.

`warnings.warn` consults the process-global `warnings.filters` list and the
per-module `__warningregistry__` dict. Both are mutable globals.

CPython's `_warnings` module guards filter list reads and `__warningregistry__`
dict writes with internal locks; under free-threading these have been audited
upstream. So concurrent warning emission is expected to be safe.

**Investigation question:** is anything in Triton resetting
`warnings.filters` (e.g. `warnings.simplefilter("always")` in tests) while
compilation is in progress? If yes, the `__warningregistry__` cache key hash
could observe inconsistent state. Almost certainly not on production paths,
but worth a one-line grep.

### 5. `JITFunction._src` rebind during `parse()`

In `JITFunction.parse`:

```python
def parse(self):
    tree = ast.parse(self._src)
    assert isinstance(tree, ast.Module)
    ...
```

`self._src` is a string set in `JITFunction.__init__` and may be replaced by
`_unsafe_update_src`. Strings are immutable, so a single attribute load gives
a stable snapshot. **No race in the read itself.**

But: `parse()` is called from compile-path code (`code_generator.py`
`call_JitFunction` / `visit_FunctionDef`) and also from `JITFunction.cache_key`
(under `_hash_lock`). If thread A is mid-compile parsing the old `_src` while
thread B does `_unsafe_update_src`, the two threads see two different ASTs.
The cache key was computed against whichever `_src` was visible at hash time
— if a swap happens between hash and parse, the compiled kernel would be
keyed under the old hash but contain code from the new source.

**Investigation question:** is `_unsafe_update_src` actually called outside
unit tests? The docstring says "only used by tests." If so, downgrade to
"not worth reporting." If any production path uses it (e.g. autotune-driven
recompile), this is a real Tier 2 issue and the `parse()` site is the
observable point.

### 6. Builder attribute storage (`codegen_fns` / `module_map` / `options`)

In `CodeGenerator.__init__`:

```python
self.builder.options = options
self.builder.codegen_fns = codegen_fns
self.builder.module_map = {} if module_map is None else module_map
```

`self.builder` is `ir.builder(context)` — a pybind11-bound C++ class
(`python/src/ir.cc`). Triton sets arbitrary Python attributes on it, then
reads them back during AST visitation. Whether these go into a per-instance
Python `__dict__` (slot-free pybind11 type) or into a typed C++ field depends
on the pybind11 binding declaration.

Per-instance `__dict__` writes are per-call (each `CodeGenerator` constructs a
fresh `ir.builder`), so cross-thread sharing only matters if two threads share
a `CodeGenerator` instance — which they do not in normal flow.

**Investigation question (for Pass 3 native review):** does
`py::class_<ir::builder>(...)` declare a `dynamic_attr()` storage (Python
`__dict__`-backed) or a fixed `def_readwrite` field? If the latter, the
attribute store is a direct C++ pointer write with no Python-level locking.
Under free-threading, that's a torn-pointer hazard if two threads ever share
the builder. Mainly a robustness audit point — current usage doesn't share.

### 7. Lazy intra-function imports of the semantic classes

In `CodeGenerator.__init__`:

```python
if is_gluon:
    from triton.experimental.gluon.language._semantic import GluonSemantic
    self.builder = gluon_ir.GluonOpBuilder(context)
    self.semantic = GluonSemantic(self.builder)
else:
    from triton.language.semantic import TritonSemantic
    self.builder = ir.builder(context)
    self.semantic = TritonSemantic(self.builder)
```

These imports run **per `CodeGenerator.__init__` call**, not at module import
time. The first-ever call into either branch triggers the actual import, which
acquires the import lock and initializes `triton.language.semantic` /
`triton.experimental.gluon.language._semantic`.

CPython's import lock is held even under free-threading, so the first-import
race is not a concern. Subsequent imports are dict lookups on `sys.modules`,
which is safe under free-threading.

**Risk angle worth checking:** does either of those modules execute
non-idempotent module-level work that could race a concurrent first-use from
another thread *before* the import lock is acquired (e.g. side effects in
already-imported sub-modules)? Module init runs under the import lock, so
unlikely. Listed for completeness.

### 8. Backend-supplied `module_map` / `codegen_fns` identity across `compile()` calls

NVIDIA `get_codegen_implementation` (in `third_party/nvidia/backend/compiler.py`)
and `get_module_map` both return **fresh dict literals** per call. AMD does
the same (`third_party/amd/backend/compiler.py`). So those dicts are not
shared across compile invocations.

But the dict **values** are shared — `cuda.convert_custom_float8_sm80`,
`min_dot_size(self.target)`, `libdevice` module — all are process-global. If
any of those values are mutable and code_generator inspects their attributes
during compilation, that's the cross-call shared-state surface.

**Investigation question:** are any of the codegen-supplied callables backed
by mutable closures, or do they read mutable module-level state during their
execution? `cuda.convert_custom_float8_sm80` and `min_dot_size` should be
pure functions; verify with a grep into `triton.language.extra.cuda`.

### `make_launcher.py` — empty file

`compiler/make_launcher.py` is a 0-byte file (`wc -l` reports 0). The actual
launcher generation lives in backend-specific files (`runtime/build.py`,
backend `driver.py` files). No content to audit in this component.

## Out of scope for this pass

The following are intentionally not investigated here:

- MLIR / `ir.builder` / `ir.context` thread-safety internals. Pass 3
  (`python/src/ir.cc`) covers these.
- `JITFunction.get_capture_scope` / `JITFunction.cache_key` /
  `JITFunction.parse` — `runtime/jit.py` audit responsibility, except where
  they are reached *from* `code_generator.py` (referenced in items #1 and
  #5 above).
- Backend `make_ttir` / `make_ttgir` pipeline entry points beyond
  `code_generator.ast_to_ttir` — these run after AST→IR lowering and belong
  to the backend audit.
- Gluon-specific lowering path. The `is_gluon` branches in `__init__` and
  the `from triton.experimental.gluon...` imports are noted but a full
  gluon audit is the `experimental/` pass.
