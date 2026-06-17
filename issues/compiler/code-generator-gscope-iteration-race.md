# `CodeGenerator.__init__` iterates live `fn.__globals__`

- **Issue-Id:** FT009
- **Status:** Open — analysis done, patch to be developed
- **Severity:** MED
- **Component:** `python/triton/compiler/code_generator.py`
- **Tier:** 1

- **Shared state:** The defining module's `__dict__`, reached as
  `fn.__globals__` via `JITFunction.get_capture_scope()` in
  `runtime/jit.py`, which returns the live dict by identity in the
  common no-closure case.
- **Writer(s):** Any thread doing module-level assignment in the
  kernel's defining module.
- **Reader(s):** `CodeGenerator.__init__` iterates `gscope.items()`
  during `ast_to_ttir` (compile hot path). Repeated per callee in
  `visit_Call`.
- **Race scenario:** Thread A compiles a kernel; `gscope` is
  `fn.__globals__` by identity. Thread B assigns a module-level name
  in the same module. Iteration under concurrent mutation has undefined
  element sequence (per `PYTHON_THREADSAFETY.md`): a legitimate global
  may be skipped → spurious `NameError` at compile time, or a
  `constexpr` global is observed post-rebind while the cache key was
  computed against the old value → stale-keyed compiled kernel.
- **Suggested fix:** Two parts. (1) Snapshot the scope —
  `scope = dict(self.__globals__)` — so codegen iterates a call-local dict and
  never touches the live module `__dict__` (kills the iteration UB). (2) Pin
  this function's *own direct* references (data, modules, **and JITCallables**)
  to the values the cache key was computed against, so codegen bakes in exactly
  what the key encodes. **Do not** build that overlay from
  `self.used_global_vals` — it is transitive, `(name, id)`-keyed, and omits
  JITCallables, so it both clobbers cross-module names and misses the
  JITCallable rebind race. The robust fix adds a separate per-function,
  non-transitive capture map filled by `DependenciesFinder`. See the analysis
  below. **No patch is committed yet** — to be developed.

## Fix analysis (FT009 working notes)

*Analysis only, no fix committed yet.* These notes evaluate candidate fixes and
explain why the previously-drafted `used_global_vals` overlay (now removed) was
wrong.

### Context / where the code lives

- `JITCallable.get_capture_scope()` — `triton/python/triton/runtime/jit.py:507`
- `JITCallable.cache_key` (computes hash, populates `used_global_vals`) — `jit.py:514`
- `DependenciesFinder` (AST walk that finds deps + records globals) — `jit.py:36`
  - `record_reference` — `jit.py:119` (JITCallables go to `_update_hash` and are **not** stored)
  - `visit_Name` → `name_lookup` — `jit.py:159`, `jit.py:167`
  - `_update_hash` merges callee `used_global_vals` (transitive) — `jit.py:101`
- launch-time "global changed since compile" check — `jit.py:760`
- Consumer: `CodeGenerator.__init__` builds `self.gscope` from the scope — `code_generator.py:309`
  - kernel entry uses `fn.get_capture_scope()` — `code_generator.py:1652`
  - **each callee** is compiled with **its own** `fn.get_capture_scope()` — `code_generator.py:1331`
  - global name resolution / "only constexpr globals" rule — `code_generator.py:373`

Compile ordering (matters for "is the capture populated yet"): `compile()` computes
`src.hash()` → `fn.cache_key` (`compiler.py:76`, `:247`) **before** `src.make_ir()` →
codegen (`compiler.py:308`). Callee `cache_key` is computed during the caller's
`DependenciesFinder` pass (`_update_hash` → `func.cache_key`), which also precedes codegen.

### The rejected change (overlay from `used_global_vals`)

This was the previously-drafted FT009 patch (now removed):

```python
def get_capture_scope(self):
    fn = self.fn
    if fn.__closure__ is None:
        scope = dict(self.__globals__)
    else:
        nonlocals = {name: cell.cell_contents
                     for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
        scope = self.__globals__ | nonlocals
    for (name, _), (val, _) in self.used_global_vals.items():   # <-- the problem
        scope[name] = val
    return scope
```

The `dict(self.__globals__)` snapshot is correct and necessary (it fixes the iteration
UB: `CodeGenerator.__init__` iterates the scope while another thread may mutate the live
module `__dict__`). **The overlay loop is the problem.**

### Why the overlay is wrong

`used_global_vals` is the wrong source for codegen pinning. Three independent facts:

1. **It is transitive.** `_update_hash` (jit.py:113) merges every callee's
   `used_global_vals` into the caller's. Its key is `(name, id(globals_dict))`, value is
   `(val, globals_dict)`.

2. **The overlay is name-keyed but the map is `(name, id)`-keyed.** So `scope[name]=val`
   pulls in bindings that belong to *callees*, including callees **in other modules**, and
   writes them over this function's own `name`. → the original cross-module clobber bug.

3. **It deliberately omits JITCallables.** `record_reference` (jit.py:140) sends a
   JITCallable to `_update_hash` and returns **before** storing it in `used_global_vals`.
   So no overlay built from `used_global_vals` can ever pin a JITCallable.

### Why the proposed `globals_dict is self.__globals__` filter is only partial

```python
for (name, globals_id), (val, globals_dict) in self.used_global_vals.items():
    if globals_dict is self.__globals__:   # identity compare, better than id()
        scope[name] = val
```

- **Fixes:** the cross-*module* clobber (other-module callees have a different
  `__globals__`, so they're filtered out).
- **Still broken — same-module transitive bleed:** a same-module callee shares
  `id(__globals__)`, so its bindings still pass the filter. The filter means "bindings
  from this module used by this function *or any transitive callee*," not "bindings used by
  this function."
- **Still broken — closure/nonlocal handling:** `DependenciesFinder` stores nonlocals
  against the temporary `inspect.getclosurevars(...).nonlocals` dict, not
  `self.__globals__`. Filtering on `self.__globals__` drops those bindings. And if pinned
  globals are overlaid *after* `self.__globals__ | nonlocals`, a pinned module global can
  override a closure var of the same name. (Partial mitigation would have to apply pinned
  globals *before* nonlocals.)
- **Still broken — JITCallable race:** unaffected, because JITCallables aren't in
  `used_global_vals` at all:

  ```python
  helper = helper_v1

  @triton.jit
  def kernel(x):
      return helper(x)
  ```

  Thread A computes the cache key from `helper_v1.cache_key`. Thread B rebinds
  `helper = helper_v2`. Codegen snapshots globals later and resolves `helper_v2`. The cache
  key encodes v1; the compiled kernel is built against v2 → stale-keyed miscompile.

So filtering is: **correct for the live-dict iteration race, partial for same-module
stale data globals, still incomplete for stale-keyed codegen correctness.**

### Key realization

`used_global_vals` is a **transitive, deep-copied change-detection baseline** whose job is
the launch-time validation at `jit.py:760`. Codegen pinning needs a *different* thing:
**this one function's own, direct, name→value snapshot, including JITCallables**, frozen to
the values the cache key was computed against. The review feedback ("capture globals once
and use the same snapshot for dependency hashing and codegen") is really asking for that
second structure.

Also note the consumer shape: every callee is compiled with **its own**
`get_capture_scope()` (`code_generator.py:1331`), so the capture only ever needs this
function's *direct* references — never the transitive set.

### Robust fix: a second, purpose-built capture map

Keep `used_global_vals` exactly as-is. Add a **per-function, non-transitive** dict that
`DependenciesFinder` fills during the same AST walk, capturing every name this function
resolves from its own globals/nonlocals — **including JITCallables and modules**.

```python
# DependenciesFinder.__init__
self.captured_refs: Dict[str, Any] = {}        # name -> value at hash time

# DependenciesFinder.visit_Name, right after `val, var_dict = name_lookup(node.id)`:
if var_dict is not None:
    self.captured_refs[node.id] = val          # JITCallables, modules, constexprs, data
```

Persist it on the function next to `used_global_vals` (around `jit.py:528`):

```python
self._captured_refs = dict(dependencies_finder.captured_refs)
```

Overlay *that* per-function map (name-keyed, so no foreign entries possible, and it carries
the JITCallable):

```python
def get_capture_scope(self):
    fn = self.fn
    scope = dict(self.__globals__)                       # snapshot -> kills iteration UB
    if fn.__closure__ is not None:
        nonlocals = {name: cell.cell_contents
                     for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
        scope.update(nonlocals)
    scope.update(self._captured_refs)                    # pin this fn's own bindings (incl. JITCallables)
    return scope
```

This closes all four gaps simultaneously:

| Gap | Closed by |
|-----|-----------|
| iteration UB on live `__globals__` | `dict(self.__globals__)` snapshot |
| cross-module clobber | per-function map, name-keyed (no foreign entries) |
| same-module transitive bleed | the new map is **not** transitive |
| `helper_v2` JITCallable race | JITCallables are now captured and pinned |

### Open design decisions (A–D)

Recommendations in **bold**; none committed.

**A. Overlay onto `dict(self.__globals__)`, or *replace* globals with the snapshot?**
Recommend **overlay** (code above). A pure replace is the literal "one snapshot" reading
but risky: `DependenciesFinder` and `CodeGenerator` walk the same AST with *different*
visitors, and it's not proven the dep-finder resolves a superset of the names codegen's
`global_lookup` (`code_generator.py:373`) resolves. A missed name becomes a fresh
`NameError` regression. Overlay keeps full globals as fallback (missed names stay unpinned
— today's behavior) while pinning everything captured. Strictly safer, fixes every known
race.

**B. Deepcopy data values in `captured_refs`, or store live refs?**
`used_global_vals` deepcopies (immune to in-place mutation, for the launch check). For
codegen-vs-key consistency on a `constexpr` baked into IR, **reuse the same deepcopy for
data values and store JITCallables/modules by reference** (identity matters; they aren't
deepcopy-friendly). I.e. populate `captured_refs` data entries to point at the same object
`used_global_vals` stored.

**C. Latent nonlocal-precedence inconsistency — fix now or just note it?**
`DependenciesFinder.name_lookup` checks **globals before nonlocals** (jit.py:167-173), but
real Python (and codegen's `__globals__ | nonlocals`) has the **freevar shadow the
global**. So a name that is both a module global and a closure var is *hashed* as the
global but *codegen'd* as the nonlocal. The unified capture only papers over this if we
capture with correct precedence. Recommend **fix `name_lookup` to check nonlocals first**
as part of this (it's the root of the item-2 ordering hazard). Small behavior change —
flag if it should be kept out of FT009.

**D. Population timing.**
`get_capture_scope` must run after `cache_key`. It does today for both kernel and callees
(see ordering note above). To be safe against future call orders, have
`get_capture_scope` touch `self.cache_key` first (cached, idempotent) so `_captured_refs`
is guaranteed populated, instead of silently returning unpinned globals.

**Recommended combination:** A = overlay, B = reuse deepcopy + JITCallables by ref,
C = fix the precedence, D = guard via `cache_key`. Smallest change that is actually
*correct* rather than partial.

### Narrow partial alternative (if a robust fix is too big right now)

```python
def get_capture_scope(self):
    fn = self.fn
    scope = dict(self.__globals__)
    for (name, _), (val, globals_dict) in self.used_global_vals.items():
        if globals_dict is self.__globals__:
            scope[name] = val
    if fn.__closure__ is not None:
        nonlocals = {name: cell.cell_contents
                     for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
        scope.update(nonlocals)            # nonlocals applied AFTER pins (item-2 ordering)
    return scope
```

Describe honestly as: correct for the live-dict iteration race; partial for same-module
stale data globals; **still incomplete** for full stale-keyed codegen correctness (does
not pin JITCallables). Not a substitute for the robust fix.
