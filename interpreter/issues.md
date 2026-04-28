# interpreter Free-Threading Issues

Initial (shallow) pass. Each triage-note entry is a candidate intended for a
follow-up deep dive — severity/tier assignments are preliminary.

Files in scope:

- `python/triton/runtime/interpreter.py` — CPU interpreter fallback for
  `@triton.jit` kernels (activated by `knobs.runtime.interpret` /
  `TRITON_INTERPRET=1`). Hosts `InterpreterBuilder`, `GridExecutor`,
  `FunctionRewriter`, `InterpretedFunction`, and the `_patch_lang` monkey-
  patch machinery that swaps `tl.*` symbols to interpreted equivalents.
- `python/src/interpreter.cc` — pybind11 bindings for `load`, `store`,
  `atomic_rmw`, `atomic_cas` used by the interpreter. Atomic fallbacks are
  serialized via a single namespace-level `std::mutex atomic_op_guard`.

## Issues

| # | Severity (prelim) | Component | Tier | Issue |
|---|-------------------|-----------|------|-------|
| 1 | SEVERE | `interpreter_builder` | 2 | [Global `interpreter_builder.grid_idx` / `grid_dim` trampled by concurrent kernel runs](builder-grid-state-race.md) |
| 2 | SEVERE | `_patch_lang` / `_LangPatchScope` | 2 | [Process-wide monkey-patching of `tl.*` with save/restore permanently corrupts `tl` under concurrent runs](patch-lang-save-restore-race.md) |
| 3 | Significant | `FunctionRewriter._compile_and_exec` | 2 | [Mutation of user kernel's `fn.__globals__` during `exec`, shared across threads](compile-and-exec-globals-mutation.md) |
| 4 | Significant | `InterpretedFunction.__call__` | 2 | [`__call__` path applies `_patch_lang` without ever restoring, leaking patches](interpreted-fn-call-leaks-patches.md) |
| 5 | Minor | `InterpretedFunction.rewritten_fn` | 2 | [Class-level rewrite cache is check-then-set; duplicate AST rewrite under concurrent first-use](rewritten-fn-cache-toctou.md) |
| 6 | Minor | `_patch_lang` | 2 | [Iteration over `fn.__globals__.items()` while `_compile_and_exec` mutates the same dict](patch-lang-globals-iteration.md) |
| 7 | Minor | `interpreter.cc` `mem_semantic_map` | 1 | [`std::map` read via non-const `operator[]` on the hot path (insert-on-miss latent risk)](mem-semantic-map-operator.md) |

`atomic_op_guard`-based serialization in `interpreter.cc` is already
documented as protected state (see CLAUDE.md and `specialize/` notes). It is
not re-reported, but see the triage notes for a scalability comment.

## Triage notes

### 1. Global `interpreter_builder.grid_idx` / `grid_dim`

- **Code (`interpreter.py:775-777`):** module-level singletons shared by
  every interpreter invocation process-wide:
  ```python
  interpreter_builder = InterpreterBuilder()
  interpreter_semantic: TritonSemantic = TritonSemantic(interpreter_builder)
  ```
- **Writers (`interpreter.py:292-302, 1290-1295`):**
  `InterpreterBuilder.set_grid_dim` / `set_grid_idx` write
  `self.grid_dim = (nx, ny, nz)` and `self.grid_idx = (x, y, z)`.
  `GridExecutor.__call__` drives them from its grid loop:
  ```python
  interpreter_builder.set_grid_dim(*grid)
  for x in range(grid[0]):
      for y in range(grid[1]):
          for z in range(grid[2]):
              interpreter_builder.set_grid_idx(x, y, z)
              self.fn(**args)
  ```
- **Readers (`interpreter.py:406-412`):**
  `create_get_program_id(axis)` / `create_get_num_programs(axis)` read
  `self.grid_idx` / `self.grid_dim` — these are what `tl.program_id()` and
  `tl.num_programs()` return inside an interpreted kernel.
- **Race:** Two threads running interpreter kernels concurrently both
  overwrite `interpreter_builder.grid_idx`. Thread A writes `(5, 0, 0)`,
  Thread B writes `(12, 0, 0)` before Thread A's kernel body reads it;
  Thread A's kernel then observes `program_id(0) == 12`. This is silent
  **wrong kernel execution** — each thread sees a mixture of both grids'
  program ids.
- **Deep-dive questions:**
  - Is the interpreter intended to support concurrent use at all? (It is a
    debugging path, but nothing at the entry point enforces single-thread.)
  - Fix direction: move per-call state (`grid_idx`, `grid_dim`, arguably the
    whole builder) into a `ContextVar` or thread-local, or serialize the
    whole `GridExecutor.__call__` with a process-wide lock.
- **Tier 2**, SEVERE.

### 2. `_patch_lang` save/restore on shared module objects

- **Code (`interpreter.py:780-797, 800-811, 814-834, 1032-1045, 1048-1125, 1130-1142`):**
  `_LangPatchScope.set_attr` reads `getattr(obj, name, _MISSING)` and then
  `setattr(obj, name, value)` on **shared Python module/class objects**
  — `tl`, `tl.core`, `tl.tensor`, `tl.dtype`, `tl.math`,
  `tl.core.tensor_descriptor_base`. `restore()` walks `_changes` in reverse
  and setattr's the saved "originals" back.
- **Writer(s):** every `GridExecutor.__call__` enters via
  `patch_scope = _patch_lang(self.fn)` at line 1280, and `restore()` fires
  in a `finally` at 1301-1302. Each call mutates globals like `tl.range`,
  `tl.static_assert`, `tl.reduce`, `tl.associative_scan`,
  `tl.tensor.__bool__`, `tl.dtype.to_ir`, every `tl.core.is_builtin(...)`
  member of `tl`, `tl.core`, `tl.math`, etc.
- **Race (permanent corruption):**
  - T1 enters `_patch_lang`, reads original `tl.range` → `R0`, sets
    `tl.range = R1`.
  - T2 enters `_patch_lang`, reads "original" `tl.range` → **`R1`**
    (T1's patched value), sets `tl.range = R2`.
  - T1 exits, restores `tl.range = R0`.
  - T2 exits, restores `tl.range = R1`. **The module is now pinned to the
    patched-interpreter version** for all non-interpreter threads, because
    T2 captured the wrong "original".
  - Worse variant: T1 reads `R0`, T2 reads `R0`, T1 sets `R1`, T2 sets `R2`,
    T1 restores `R0`, T2 restores `R0` (restored twice, benign this run,
    but during the window T1 is running its kernel body with T2's `R2`
    installed → wrong-semantic execution for the rest of T1's run).
- **Additional damage:** the class/module attributes being written include
  `tensor.__bool__`, `tensor.__repr__`, `tensor.__index__`, and a `T`
  property descriptor. These affect every `tl.tensor` instance process-wide
  (including non-interpreter threads). A leak here means GPU-path threads
  see interpreter-only `__bool__` semantics.
- **Deep-dive questions:**
  - Is the "interpreter is single-threaded by design" assumption explicit
    anywhere? If so, a doc note + assertion is a cheap partial fix.
  - Full fix likely requires routing `tl.*` dispatch through a
    `ContextVar` / thread-local "semantic" rather than monkey-patching
    shared module state. That is a significant architectural change.
- **Tier 2**, SEVERE.

### 3. `FunctionRewriter._compile_and_exec` mutates `fn.__globals__`

- **Code (`interpreter.py:1382-1390`):**
  ```python
  def _compile_and_exec(self, transformed_ast):
      compiled_code = compile(transformed_ast, filename=self.filename, mode='exec')
      local_namespace = {**self.kwargs}
      fn_globals = self.fn.__globals__
      for key, value in globals().items():
          if key not in fn_globals:
              fn_globals[key] = value
      exec(compiled_code, fn_globals, local_namespace)
      return local_namespace[self.fn.__name__]
  ```
- **Shared state:** `self.fn.__globals__` is the user-supplied kernel's
  module globals. It is shared across every thread that calls the same
  kernel.
- **Writer(s):** any thread triggering `FunctionRewriter.rewrite_ast`. The
  cache at `InterpretedFunction.rewritten_fn` (see #5) is check-then-set, so
  under concurrent first-use the rewrite runs in parallel for the same
  function, with both threads copying the interpreter module's globals into
  the same `fn_globals` dict.
- **Reader(s):** the kernel itself (when executed), plus any other code in
  the kernel's module that reads its own module globals.
- **Race consequences:**
  - Concurrent `fn_globals[key] = value` for the same key is a per-key
    atomic setitem — benign on its own (matches the `feedback_dict_threadsafe`
    memory).
  - `if key not in fn_globals: fn_globals[key] = value` is a check-then-act
    TOCTOU — two threads both pass the check and both write the same value.
    Benign.
  - `exec(compiled_code, fn_globals, ...)` however runs the freshly compiled
    body with `fn_globals` as the globals dict, which CPython's byte-code
    reads during execution. A concurrent writer still copying keys into
    `fn_globals` mutates the dict while `exec` iterates it as a namespace.
    Lookups are individually atomic, but a kernel body that uses a module
    global set by the copy loop (e.g. `interpreter_builder`,
    `interpreter_semantic`, `np`, `math`) can race with the copy that
    installed those names. In practice the copy typically completes before
    the AST transformer's generated references resolve, but there is no
    synchronization guaranteeing this.
  - **Side effect persists:** after the race, the user's kernel module
    permanently contains interpreter-internal globals like
    `interpreter_builder`, `interpreter_semantic`, `_implicit_cvt`, etc.
    That is already a surprising behavior; it is exacerbated under
    concurrent first-use because the dict mutation is unsynchronized.
- **Deep-dive questions:**
  - Is the `fn_globals` injection strictly needed? The AST transformer
    references `interpreter_semantic` by name, so `exec` needs to see it —
    but it could be provided via the `globals` arg to `exec` (a merged view)
    instead of mutating the kernel module's globals.
  - Does CPython 3.14t's dict guarantee forward progress for `exec`
    readers while another thread is `__setitem__`-ing? (Believed yes per
    PEP 703, but verify.)
- **Tier 2**, Significant (correctness likely survives via single-use caching
  but the shared-state mutation is unsafe and leaks into the user's module).

### 4. `InterpretedFunction.__call__` leaks patches

- **Code (`interpreter.py:1425-1432`):**
  ```python
  def __call__(self, *args, **kwargs):
      # This is a device function call
      _patch_lang(self.fn)
      fn = self.rewrite()
      try:
          return fn(*args, **kwargs)
      except Exception as e:
          raise InterpreterError(repr(e)) from e
  ```
- **Concern:** unlike `run()` / `GridExecutor.__call__` (which stores
  `patch_scope` and calls `patch_scope.restore()` in a `finally`),
  `__call__` **throws away the returned `_LangPatchScope`**. Every device-
  function call permanently leaks all `tl.*` patches into the process.
- **Free-threading angle:** under single-threaded GIL use, the effect is
  that the "outer" `GridExecutor.__call__`'s restore eventually reverts
  everything (because the outer scope captured originals before the nested
  call further patched). Under free-threading it is worse: the nested
  `_patch_lang` call on T1 captures T2's live patches as "originals" (see
  issue #2) and never restores them anyway. The two issues reinforce each
  other.
- **Deep-dive questions:**
  - Is this intentional (nested kernel call, rely on outer restore)?
    If so, skip the inner `_patch_lang` entirely — the outer scope already
    installed the patches.
  - Is `InterpretedFunction.__call__` reachable outside a `GridExecutor`
    context? If so, the leak is immediate even under the GIL.
- **Tier 2**, Significant.

### 5. `InterpretedFunction.rewritten_fn` class-level cache

- **Code (`interpreter.py:1393-1419`):**
  ```python
  class InterpretedFunction(KernelInterface[T]):
      rewritten_fn: Dict[Callable, Callable] = {}
      ...
      def rewrite(self):
          if self.fn not in self.rewritten_fn:
              self.rewritten_fn[self.fn] = self.rewriter.rewrite_ast()
          return self.rewritten_fn[self.fn]
  ```
- **Concern:** class attribute shared by *every* `InterpretedFunction`
  instance, i.e. one dict per process. Classic check-then-act: two threads
  calling the same kernel for the first time both pass the `not in` guard
  and both run `rewrite_ast()` (which calls `inspect.getsourcelines`, runs
  the `ASTTransformer`, mutates `fn.__globals__` — see #3 — and `compile`s
  the AST). Last writer wins.
- **Classification:** duplicate compile/rewrite. Benign in correctness terms
  (both rewrites produce the same function object modulo identity), but
  couples directly to #3's globals mutation.
- **Deep-dive questions:**
  - Can `FunctionRewriter.rewrite_ast` produce different results on
    different threads (e.g., different `JITFunction` instantiation leading
    to different line numbers)? Probably no.
  - Fix direction: per-kernel lock, or `functools.lru_cache`-style coalescing
    keyed on `self.fn`.
- **Tier 2**, Minor.

### 6. `_patch_lang` iterates `fn.__globals__.items()`

- **Code (`interpreter.py:1130-1142`):**
  ```python
  def _patch_lang(fn):
      scope = _LangPatchScope()
      langs = [value for _, value in fn.__globals__.items()
               if inspect.ismodule(value) and value in [tl, tl.core]]
      ...
  ```
- **Concern:** `fn.__globals__` is the kernel's module globals — the same
  dict `FunctionRewriter._compile_and_exec` mutates via the copy loop at
  `interpreter.py:1386-1388`. If T1 is in `_compile_and_exec` copying keys
  while T2 is in `_patch_lang` iterating, T2's iteration observes an
  undefined element sequence (see `PYTHON_THREADSAFETY.md` and
  `feedback_iteration_not_safe.md`).
- **Consequence:** worst case, T2's `langs` list misses or duplicates a
  module reference, so `_patch_builtin`/`_patch_lang_core` skip patching
  one of `tl` / `tl.core`, and the subsequent kernel body invokes
  un-patched versions of `range`, `static_assert`, `reduce`, etc. That
  yields an `InterpreterError` or silently wrong results.
- **Likelihood:** realistic only on first-ever call (the only time
  `_compile_and_exec` mutates `fn_globals`), and the mutation window is
  small. Still a real window.
- **Tier 2**, Minor.

### 7. `mem_semantic_map` read via non-const `operator[]`

- **Code (`interpreter.cc:33-38, 673-674, 720-721`):**
  ```cpp
  std::map<MemSemantic, std::memory_order> mem_semantic_map = { ... };
  // In atomic_rmw / atomic_cas bindings:
  std::memory_order order = mem_semantic_map[sem];
  ```
- **Concern:** `mem_semantic_map` is populated at namespace-scope init time
  (before any thread can reach the bindings, protected by the import lock).
  It is thereafter read-only **in practice**. But `std::map::operator[]` is
  non-const and inserts a default-constructed value on missing key. If a
  future maintainer extends `MemSemantic` without updating the map, two
  threads both reach the binding with the missing key and both try to
  insert — that is concurrent mutation of a non-thread-safe container
  (SEVERE if it ever happens).
- **Classification today:** Minor / latent-hazard. The current enum values
  are all pre-inserted, so there is no *present* bug — only a footgun.
- **Deep-dive questions:**
  - Switch to `at(sem)` (throws on miss) or a `switch` / `constexpr` map.
  - Or mark the global `const` so the non-const `operator[]` overload is
    not available.
- **Tier 1**, Minor.

## Triage notes, continued

### `atomic_op_guard` scalability note (not reported)

`interpreter.cc:21` declares a single process-wide `std::mutex atomic_op_guard`
used by every fallback path in `atomic_fadd<npy_half>`, `atomic_cmp`, and all
`AtomicRMWOp<DType, Op>` branches where
`is_reinterpret_cast_to_atomic_safe<T>` is false. Correctness is fine (this is
the "already-protected state" carve-out in CLAUDE.md). It is however a **single
global lock** — concurrent fallback atomics on disjoint buffers serialize on
it. Not a bug; flagged as follow-up performance work if the interpreter is
ever positioned for parallel workloads.

### Write-once / not worth reporting

- **`InterpreterBuilder.ir_sem_to_interpreter_sem` / `ir_rmw_op_to_interpreter_rmw_op`**
  class attributes (lines 265-283). Populated at class-definition time; never
  mutated. Read-only.
- **`InterpreterBuilder.codegen_fns`** (lines 288-290). Populated once in
  `__init__`. Never mutated after module init.
- **`FunctionRewriter.ast_transformer = ASTTransformer()`** (line 1324). One
  shared instance used by every `_transform_ast` call; `ASTTransformer` holds
  no per-call state in its `visit_Assign`, so concurrent use is OK.
- **`np_erf_fp32`, `np_erf_fp64`, `np_umulhi_u64`** (lines 252-254). Module-
  level `np.vectorize` wrappers; treated as pure functions.
- **`_MISSING = object()`** sentinel (line 775). Immutable.
- **`mem_semantic_map`, `atomic_op_guard`** initialization at C++ namespace
  scope — covered by import-lock carve-out.
- **`CompileTimer`-style state, caches on `JITFunction`, etc.** — out of
  scope for this component; see `jit/` and `autotuner/`.

## Caller-side concerns noted but out of scope

- **`knobs.runtime.interpret` is a Tier-3 config flag** that selects the
  interpreter path in `jit.py:967-968` and `1146-1147`. Covered by the
  `knobs/` component. A mid-run toggle would be a cross-component concern.
- **`JITFunction(self.fn)` is constructed inside
  `FunctionRewriter._get_jit_fn_file_line`** (line 1355) solely to read
  `file_name` / `def_file_line_number`. That implicitly registers the jit
  function in `_triton_jit_function_registry` — already covered under the
  `jit/` component.
