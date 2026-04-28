# language

Scope: `python/triton/language/` — the Triton DSL surface (`core.py`,
`semantic.py`, `standard.py`, `math.py`, `random.py`, `target_info.py`,
`extra/__init__.py`, `extra/libdevice.py`, top-level `__init__.py`).

This is a **scoping pass**. The goal is to enumerate potential free-threading
concerns for a later deep dive — not to fully adjudicate each one.

## Overall character

The `language/` package is overwhelmingly **declarative**:

- Most user-facing code consists of `@jit`, `@builtin`, `@constexpr_function`,
  or `@_tensor_member_fn`-decorated functions. These are called *during*
  kernel compilation (which has its own concurrency story covered by `jit`,
  `compiler`, and `specialize` audits), not persistently mutated across calls.
- `semantic.py`'s `TritonSemantic` class is instantiated per-compile with a
  fresh `ir.builder`. It has no module-level mutable state. Its instance
  fields (`builder`, `tensor`, `lang`) are set once in `__init__`.
- `math.py`, `random.py`, `standard.py`, `extra/libdevice.py` are pure
  `@jit`-decorated function bodies and stub declarations — no module state.

The interesting surface is narrow: a few module-level singletons, one
`@cached_property`, a couple of import-time class mutations, and a
`sys.modules` mutation in `extra/__init__.py`.

## Issues

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| 1 | Minor | `language/core.py` | `tuple_type.name` uses `@cached_property` — first-use write to instance `__dict__` on potentially shared `tuple_type` instances |
| 2 | Minor | `language/core.py` | `dtype` singletons (`int32`, `float32`, …) are instance objects with attributes set in `__init__`; verify they remain read-only after import |
| 3 | Minor | `language/core.py` | `_tensor_member_fn` and similar decorators call `setattr(tensor, ...)` at import time — confirm no post-import mutation |
| 4 | Minor | `language/core.py` | `_aggregate()` dynamically builds `aggregate_value` classes at decorator application time; instances have mutable `__dict__` populated in `__init__` / `_unflatten_ir` |
| 5 | Minor | `language/target_info.py` | `current_target()` reads `driver.active` with no synchronization; overlaps with `runtime-driver` audit but worth confirming |
| 6 | Minor | `language/extra/__init__.py` | Module-body loop appends to `_backends` list and writes to `sys.modules[module_name]` during subpackage discovery; relies on Python's import lock |

## Triage notes

### 1. `tuple_type.name` @cached_property (core.py:767)

```python
class tuple_type(base_type):
    def __init__(self, types, fields=None):
        self.types = types
        self.fields = fields

    @cached_property
    def name(self):
        ...
```

The pattern: a `tuple_type` instance is created (often during IR construction
or type lookup via `str_to_ty`), and `name` is computed lazily on first
access. `@cached_property` writes the result into the instance's `__dict__`.

Concern: if the **same** `tuple_type` instance is shared across concurrent
compiles (e.g., interned in some cache, or reachable from a shared registry),
two threads can race on the first-use write. On free-threaded CPython a
first-use `__dict__` insertion of an already-computed-to-same-value result
is typically benign (the value is deterministic and dict setitem is
internally safe), but `functools.cached_property` calls the wrapped function
twice in the worst case.

Deep dive needed:
- Are `tuple_type` instances interned/cached anywhere (e.g., `str_to_ty`
  returns a fresh one each call on lines 293, 324, 326)?
- Are they attached to long-lived objects that outlive a single compile?
- Does `tuple_type.__eq__` / `__hash__` make coincidental sharing possible
  through some cache elsewhere (jit / specialize)?

If instances are per-compile and not shared across threads, this is a
non-issue. If they are shared, it falls under Tier 2 and is at most a
duplicate-computation minor bug.

### 2. `dtype` singletons (core.py:807–825)

```python
void = dtype('void')
int32 = dtype('int32')
float32 = dtype('fp32')
...
```

These are constructed once at import time. `dtype.__init__` sets attributes
like `name`, `primitive_bitwidth`, `itemsize`, `int_signedness`,
`int_bitwidth`, `fp_mantissa_width`, `exponent_bias`. These are all set
during `__init__` and — based on a grep — never mutated after.

Deep dive needed: confirm no later mutation (no `setattr`/`__dict__` write)
on any of the dtype singletons. If true, this is purely read-only shared
state and safe.

Note: `dtype.to_ir(builder)` reads `self.name` and calls `builder.get_*`
methods. That's read-only on the dtype side. Any mutation concern would be
on the `ir.builder`, which is a per-compile instance.

### 3. Import-time class mutation via `_tensor_member_fn` (core.py:51–93)

`_tensor_member_fn(fn)` ends with:

```python
setattr(tensor, fn.__name__, fn if isinstance(fn, JITCallable) else wrapper)
```

This mutates the `tensor` class attribute dict at import time (each
decorator application). Under free-threading, CPython's import lock still
serializes module execution, so this is safe in practice. But if any of the
`_tensor_member_fn` decorated functions can re-register *after import* (e.g.,
via dynamic extension loading), the picture changes.

Deep dive needed: grep for any post-import callers that add tensor members
dynamically.

### 4. `_aggregate()` dynamic class construction (core.py:1587–1679)

`_aggregate(cls)` creates an `aggregate_value` class with a custom `__new__`
and `__setattr__`. Instances of this class have mutable `__dict__` filled in
by the user-defined `__init__` or by `_unflatten_ir` (line 1560:
`setattr(instance, name, value)`).

The aggregate class itself is created once per `@_aggregate` decorator
application, at import time. Instances are created per-compile and populated
before being handed out, which usually means no cross-thread sharing of a
half-initialized instance — but `_unflatten_ir` does:

```python
instance = self.base_cls._get_instance()     # bare instance
for name, ty in self.fields:
    value, cursor = ty._unflatten_ir(handles, cursor)
    setattr(instance, name, value)
return instance, cursor
```

If the same `_aggregate_type` object is used concurrently, each call creates
a fresh `instance`, so no shared instance write — this looks safe. Worth
confirming in a deep dive that `_get_instance()` really returns a new
object each call (it does: `super().__new__(this_cls)`).

### 5. `current_target()` reading `driver.active` (target_info.py:7–13)

```python
def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        return None
    return active_driver.get_current_target()
```

This is called from `@constexpr_function`-wrapped `is_cuda()`,
`cuda_capability_geq()`, `is_hip()`, etc., which are evaluated during kernel
compilation. The real concurrency story here is owned by the
`runtime-driver` audit (`DriverConfig` lazy-init and mutation).

Deep dive needed: ensure the `runtime-driver` audit covers the read paths
invoked from `current_target()`; nothing additional to add here unless
`active_driver.get_current_target()` itself has mutable state worth
flagging.

### 6. `extra/__init__.py` module-body discovery (lines 5–22)

```python
_backends = []
for module_finder, module_name, is_pkg in pkgutil.iter_modules(__path__, ...):
    ...
    spec.loader.exec_module(module)
    _backends.append(module_name)
    modules[module_name] = module
__all__ = _backends
del _backends
```

This is module-level code that runs during `import triton.language.extra`.
Python's import lock (even under free-threading) serializes module execution,
so concurrent `import triton.language.extra` calls will block rather than
race. `_backends` is deleted after assigning to `__all__`, so there is no
post-import mutable list to worry about.

The `sys.modules[module_name] = module` write is the normal module-
registration pattern. Not a concern by itself.

Deep dive needed: confirm the imported backend submodules don't start any
threads that mutate shared state before their import completes.

## Not-a-concern list (for deprioritization)

- **All `@jit`-decorated function bodies** in `math.py`, `random.py`,
  `standard.py`, `extra/libdevice.py`: these are source-level DSL code
  consumed by the compiler, not ordinary Python that runs at call time.
- **`TritonSemantic`**: per-compile instance, not shared.
- **`constexpr`, `tensor`, `pointer_type`, `block_type` classes**: attribute
  writes only in `__init__`; no post-construction mutation observed.
- **`CONSTEXPR_0 = constexpr(0)`** (core.py:356): immutable singleton.
- **`N_ROUNDS_DEFAULT = tl.constexpr(10)`** (random.py:5): immutable singleton.
- **No `@lru_cache`, `@functools.cache`, or `ContextVar` usage** in this
  package (confirmed by grep).
- **No module-level dicts/lists/sets are mutated after import** anywhere in
  `core.py`, `semantic.py`, `standard.py`, `math.py`, `random.py`, or
  `target_info.py` (confirmed by grep for module-level assignments and
  common mutation sites).

## Tier classification

All identified items are **Tier 2** (multiple threads compiling concurrently)
at most; none are on a hot dispatch path in a way that would make them
Tier 1. Severity for everything flagged here is **Minor** pending deep
investigation — the package simply doesn't have much mutable state.

## Recommended follow-up order for deep dive

1. `tuple_type.name @cached_property` — cheapest to resolve; trace whether
   `tuple_type` instances are ever shared.
2. `extra/__init__.py` import-time discovery — confirm no post-import path
   re-enters this code.
3. `dtype` singletons — grep all modules for any attribute write on the
   singleton instances (`int32.foo = ...`).
4. `_aggregate` / `aggregate_value` — confirm no instance sharing across
   compile calls.
5. `target_info.current_target()` — cross-reference against the
   `runtime-driver` audit to avoid double-reporting.
