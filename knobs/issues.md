# knobs.py Free-Threading Issues

Issues in `python/triton/knobs.py` affecting free-threaded Python 3.14t.

File in scope:

- `knobs.py` — process-global mutable configuration, hook chains, and
  callback slots. Module-level singletons `build`, `redis`, `cache`,
  `compilation`, `autotuning`, `runtime`, `language`, `nvidia`, `amd`,
  `proton` are created once at import and referenced from every hot path
  in the project (compiler, JIT dispatch, autotuner, build, launch).

Key shared objects:

- `HookChain` instances held as class attributes on `runtime_knobs`:
  `launch_enter_hook`, `launch_exit_hook`, `kernel_load_start_hook`,
  `kernel_load_end_hook`, `kernel_unload_hook`. These are **shared by
  every reader of `knobs.runtime.*_hook`** across the entire process —
  they live on the class object, not the instance.
- Plain-callable slots: `runtime.jit_cache_hook`,
  `runtime.jit_post_compile_hook`, `runtime.add_stages_inspection_hook`,
  `compilation.listener`, `build.impl`.
- Per-knob instance `__dict__` state written by the `env_base` descriptor
  on explicit assignment (`knobs.compilation.dump_ir = True`), plus
  process-global `os.environ` side effects from the same path.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | SEVERE | HookChain | 3 | [`HookChain.__call__` iterates `self.calls` under concurrent `add`/`remove`](hookchain-iteration-race.md) |
| 2 | Significant | HookChain | 3 | [`HookChain.add` / `HookChain.remove` TOCTOU on membership check](hookchain-add-remove-toctou.md) |
| 3 | Significant | runtime hook slots | 3 | [Hot-path callers re-read `knobs.runtime.*_hook` across `is not None` / call](hook-slot-caller-reread.md) |
| 4 | Significant | base_knobs.scope | 3 | [`base_knobs.scope()` corrupts shared knob state / `os.environ` under concurrent use](scope-context-manager-race.md) |
| 5 | Significant | env_base.__set__ | 3 | [`env_base.__set__` non-atomic instance-dict vs `os.environ` update](env-base-set-non-atomic.md) |
| 6 | Minor | setenv | 3 | [`setenv` check-then-delete TOCTOU on `os.environ`](setenv-toctou.md) |
| 7 | Minor | base_knobs.reset | 3 | [`base_knobs.reset()` non-atomic multi-descriptor delete](reset-non-atomic.md) |
| 8 | Minor | refresh_knobs | 3 | [`refresh_knobs()` vs concurrent readers on `runtime.debug` / `compilation.instrumentation_mode`](refresh-knobs-staleness.md) |

Cross-reference:
[`jit/add-stages-inspection-hook-toctou.md`](../jit/add-stages-inspection-hook-toctou.md)
is already open against the `knobs.runtime.add_stages_inspection_hook`
reader pattern in `jit.py`. Issue #3 here generalizes that pattern to
the rest of the hook slots (launch/kernel-load/kernel-unload/jit_cache).

## Triage notes

### 1. `HookChain.__call__` iterates `self.calls` under concurrent mutation

- **Shared state:** `HookChain.calls: list[F]` (knobs.py:411). For each
  of `launch_enter_hook`, `launch_exit_hook`, `kernel_load_start_hook`,
  `kernel_load_end_hook`, `kernel_unload_hook`, there is exactly one
  `HookChain` object installed at class definition time and reused for
  every call in the process.
- **Writer:** `HookChain.add` (knobs.py:414-416) appends; `HookChain.remove`
  (knobs.py:418-420) pops. Neither is synchronized.
- **Reader:** `HookChain.__call__` (knobs.py:422-424):
  ```python
  for call in self.calls if not self.reversed else reversed(self.calls):
      call(*args, **kwargs)
  ```
  Invoked from hot-path launch/load/unload sites in `compiler.py:442-443,
  475, 483, 511` and `jit.py:777`.
- **Race scenario:** Thread A registers a profiling hook via
  `knobs.runtime.launch_enter_hook.add(my_hook)` while Thread B is in
  the middle of `CompiledKernel.__getitem__` → native launch, which
  iterates `knobs.runtime.launch_enter_hook.calls` to drive the launch
  hook chain. Per PYTHON_THREADSAFETY.md, iteration under concurrent
  mutation of a `list` is *not* safe in free-threaded CPython: the
  interpreter will not crash, but the observed element sequence is
  undefined. A hook may be skipped or visited twice, and under
  `reversed(self.calls)` the reverse-iterator can go past the end of a
  list that has shrunk concurrently.
- **Concrete consequences:** skipped `launch_enter_hook` → unbalanced
  `launch_exit_hook` (profiling / tracing corruption); duplicate invoke
  of `kernel_load_start_hook` during concurrent kernel load; similar
  hazards in `kernel_unload_hook` during GC-triggered unloads.
- **Tier:** 3 (configuration mutation) — but note that common users
  (profilers like `triton.profiler`) mutate these chains while the
  runtime is actively launching kernels.
- **Suggested fix direction:** snapshot under a lock in `__call__`, or
  switch to an immutable tuple with CAS-style replacement on
  `add`/`remove`.

### 2. `HookChain.add` / `HookChain.remove` TOCTOU

- **Shared state:** same `HookChain.calls` list as #1.
- **Concern:** `add` does `if func not in self.calls: self.calls.append(func)`
  (knobs.py:415-416). Two threads registering the same hook concurrently
  can both see `not in` as True and both append, producing a duplicate
  entry — the hook then fires twice per launch / load. `remove` does
  `if func in self.calls: self.calls.remove(func)` (knobs.py:419-420).
  Two threads unregistering the same hook can both see `in` as True; the
  second `list.remove` raises `ValueError`.
- **Tier:** 3.
- **Severity:** Significant. The duplicate-add case silently doubles
  profiling/tracing output; the duplicate-remove case raises an exception
  on a configuration path.
- **Suggested fix direction:** lock the membership check together with
  the mutation, or accept a set-based container.

### 3. Hot-path callers re-read `knobs.runtime.*_hook`

- **Shared state:** plain-callable hook slots on the `runtime_knobs`
  singleton: `jit_cache_hook` (knobs.py:477), `jit_post_compile_hook`
  (knobs.py:480), `add_stages_inspection_hook` (knobs.py:483). Also
  applies to `HookChain`-typed slots when the reader stores a local
  reference then calls the chain — but since those are the *chain
  object* (not the calls list) and the chain object is assigned at
  class definition time, the slot value itself does not change in
  practice; the risk is only the user-replaceable bare-callable slots.
- **Reader pattern (compiler.py:442-443, 474-475, 482-483):**
  ```python
  if knobs.runtime.kernel_unload_hook is not None:
      knobs.runtime.kernel_unload_hook(self.module, ..., self.hash)
  ```
  Two attribute reads. If another thread clears the slot between the
  two, the second read is `None` and the call raises `TypeError`. If it
  swaps the slot, the launch uses a different hook than the
  `is not None` check validated.
- **Other sites to audit:** compiler.py:247-248 (same pattern on
  `add_stages_inspection_hook` during `compile()`); compiler.py:492
  (`launch_enter_hook is None`); jit.py:877, 893, 901 (`jit_cache_hook`
  / `jit_post_compile_hook`).
- **Related issue:** jit.py:727-729 has the same pattern and is already
  written up in
  [`jit/add-stages-inspection-hook-toctou.md`](../jit/add-stages-inspection-hook-toctou.md).
  The same argument applies to every compiler.py site in this section
  and should be resolved together.
- **Tier:** 3.
- **Severity:** Significant. `None()` raises on hot paths.
- **Suggested fix direction:** single-load into a local at the top of
  each reader, then operate on the local — identical to the fix
  prescribed for jit.py issue #12.

### 4. `base_knobs.scope()` context manager is not thread-safe

- **Shared state:** `self.__dict__` on the module-level knob singleton
  (for example `knobs.compilation`), plus `os.environ`.
- **Code (knobs.py:295-309):**
  ```python
  @contextmanager
  def scope(self) -> Generator[None, None, None]:
      try:
          initial_env = {knob.key: getenv(knob.key) for knob in self.knob_descriptors.values()}
          orig = dict(self.__dict__)
          yield
      finally:
          self.__dict__.clear()
          self.__dict__.update(orig)
          for k, v in initial_env.items():
              if v is not None:
                  os.environ[k] = v
              elif k in os.environ:
                  del os.environ[k]
  ```
- **Race scenarios:**
  1. *Nested/parallel `scope()` on the same singleton.* Thread A enters
     `knobs.compilation.scope()`, snapshots `orig_A`. Thread B enters
     `knobs.compilation.scope()`, snapshots `orig_B == orig_A` (because
     A has not yet mutated the dict). Both threads perform their
     overrides and yield. On exit, A runs `__dict__.clear()` +
     `update(orig_A)` — erasing every override Thread B made inside its
     own `with` block. The user's expectation that their scoped override
     is active until `__exit__` is violated.
  2. *`scope()` vs unrelated concurrent `knob.* =` assignments.*
     Outside-of-scope writers on other threads are clobbered by the
     `__dict__.clear(); __dict__.update(orig)` pair.
  3. *Concurrent `os.environ` restore.* `del os.environ[k]` races with
     similar deletes from `setenv` (issue #6).
- **Used at:**
  - `experimental/gsan/_stream_sync.py:39` —
    `with triton.knobs.compilation.scope():`
  - `tools/triton_to_gluon_translator/slice_kernel.py:317, 490`
- **Tier:** 3.
- **Severity:** Significant. A caller that uses `scope()` to temporarily
  force a compiler flag for a single compile can silently have that
  override erased by a concurrent `scope()` exit on another thread.
- **Suggested fix direction:** either document `scope()` as
  single-threaded only (it already mutates `os.environ` which is
  inherently process-global), or rework it to use a thread-local /
  `ContextVar`-based override stack. Reverting via
  `__dict__.clear(); __dict__.update(orig)` is not salvageable in a
  multi-threaded model even with a lock.

### 5. `env_base.__set__` is non-atomic across instance-dict and `os.environ`

- **Shared state:** `obj.__dict__[self.name]` on the knob singleton, and
  the global `os.environ[self.key]` entry.
- **Code (knobs.py:84-90):**
  ```python
  def __set__(self, obj, value):
      if isinstance(value, Env):
          obj.__dict__.pop(self.name, None)
      else:
          obj.__dict__[self.name] = value
          if env_val := toenv(value):
              setenv(self.key, env_val[0])
  ```
- **Race scenarios:**
  1. *Writer vs reader consulting env.* A subprocess (or a native
     extension that reads its own env) can observe the process-global
     `os.environ` updated *before or after* the Python-visible
     `obj.__dict__` is updated. The descriptor `__get__` path
     (knobs.py:75-79) prefers the Python side, so pure-Python readers
     are consistent, but any native code that reads the environment
     variable directly (for example `libtriton` reading `TRITON_DEBUG`
     via `getenv` at line 15, or external libraries reading env) sees
     a different value than Python does for a window.
  2. *Two concurrent writers.* Thread A writes `dump_ir = True` while
     Thread B writes `dump_ir = False`. Each write is a pair (dict
     write, `setenv` call) and the two pairs can interleave, leaving
     the Python value and the env string pointing at different
     underlying values.
- **Tier:** 3.
- **Severity:** Significant (for knobs whose native counterparts read
  env directly), Minor otherwise. Probably worth reporting as
  Significant because `libtriton` and backend shared objects *do* read
  env vars directly, so a Python-side `True` with an env-side `"0"` is
  a real divergence.
- **Suggested fix direction:** lock the pair, or explicitly document
  that runtime mutation of env-backed knobs is unsupported under
  free-threading and restrict mutation to import time.

### 6. `setenv` check-then-delete TOCTOU on `os.environ`

- **Code (knobs.py:32-39):**
  ```python
  def setenv(key, value):
      if not propagate_env:
          return
      if value is not None:
          os.environ[key] = value
      elif key in os.environ:
          del os.environ[key]
  ```
- **Race scenario:** Thread A enters `setenv("X", None)`, sees
  `"X" in os.environ` as True. Thread B enters `setenv("X", None)`,
  also sees True, and runs `del os.environ["X"]`. Thread A's
  subsequent `del os.environ["X"]` raises `KeyError`.
- **Tier:** 3.
- **Severity:** Minor. Only fires on a concurrent clear-to-None of the
  same env var; consequence is a raised `KeyError` from a user-facing
  knob setter.
- **Suggested fix direction:** `os.environ.pop(key, None)` instead of
  the check-then-delete pair.

### 7. `base_knobs.reset()` non-atomic multi-descriptor delete

- **Code (knobs.py:290-293):**
  ```python
  def reset(self):
      for knob in self.knob_descriptors.keys():
          delattr(self, knob)
      return self
  ```
- **Concern:** `reset()` walks every knob descriptor and deletes the
  instance override one at a time. A concurrent reader on the same
  knob singleton sees the reset as a non-atomic sequence: some knobs
  still reflect the old override, others already back to env default.
  Also races with a concurrent writer that re-sets a knob; the writer
  can be observed as "half-applied" after reset completes.
- **Tier:** 3.
- **Severity:** Minor. Only meaningful in test/tooling harnesses that
  call `reset()` while other threads are still active on the knob
  object.
- **Suggested fix direction:** lock around the whole `reset()`, or
  document single-threaded semantics.

### 8. `refresh_knobs()` vs concurrent readers

- **Code (knobs.py:574-576):**
  ```python
  def refresh_knobs():
      runtime.debug = env_bool("TRITON_DEBUG").get()
      compilation.instrumentation_mode = env_str("TRITON_INSTRUMENTATION_MODE", "").get()
  ```
- **Concern:** Both targets are plain attributes (not `env_base`
  descriptors — note that line 370 and 465 evaluate `.get()` at class
  definition time and store the scalar result), so each assignment is
  a single `obj.__dict__` write and is atomic. A concurrent reader at
  `jit.py:709-710, 724` sees either the old or new value — there is no
  intermediate state — but a reader that re-reads the value across
  multiple lines can observe a change mid-flow.
- **Tier:** 3.
- **Severity:** Minor. These two knobs are the "benign debug-flag
  staleness" carve-out from CLAUDE.md. Report only if a concrete
  correctness consequence can be established for the
  `instrumentation_mode` change (which routes to a different
  instrumentation pass list in the compiler).
- **Suggested fix direction:** if a correctness consequence is found,
  cache both values into locals at the top of the affected
  compile/launch entry points.

## Not worth reporting

- **Module-level singleton construction** (knobs.py:562-571,
  `build = build_knobs()`, `runtime = runtime_knobs()`, etc.). Written
  once at import time under the import lock, never replaced. CLAUDE.md's
  carve-out for write-once globals applies.

- **`HookChain` objects installed as class attributes on
  `runtime_knobs`** (knobs.py:468-474). The assignment
  `launch_enter_hook: HookChain[LaunchHook] = HookChain()` runs at class
  body execution time (also under the import lock). The `HookChain`
  object identity is stable for the life of the process; what races is
  the `.calls` list inside it — covered by issues #1 and #2.

- **`NvidiaTool.from_path` `@functools.lru_cache`** (knobs.py:181-190).
  `functools.lru_cache` is internally thread-safe on free-threaded
  CPython; duplicate `subprocess.check_output` invocation on a cold
  cache is benign (same stdout, same returned `NvidiaTool`).

- **C++11-style magic-static / import-time env reads.** `env_bool("...").get()`
  evaluated at class definition time (knobs.py:370, 465) happens under
  the import lock. `getenv` itself is a single C call and is concurrent-safe
  for reads.

- **`base_knobs.knob_descriptors` / `.knobs` properties**
  (knobs.py:272-283). Each returns a fresh dict on every call; callers
  cannot observe shared mutable state through them.

- **`base_knobs.copy()`** (knobs.py:285-288). Builds a new instance via
  `type(self)()` and `__dict__.update(self.__dict__)`. Two dict ops, both
  atomic; worst case `copy()` observes a partial mid-mutation snapshot
  of `self.__dict__`, which is the semantics the caller asked for.

- **`setenv` write path** (`os.environ[key] = value` on knobs.py:37).
  Single dict-assignment on `os.environ`; concurrent assignments are
  internally safe in free-threaded CPython. The hazard is the
  check-then-delete branch (issue #6).

- **Env-var-backed knob reads under `env_base.__get__`**
  (knobs.py:75-79). `obj.__dict__.get` is atomic; the fallback to
  `self.get()` hits `getenv` which reads C-level `environ`. Both
  paths are safe as reads; the hazard is on the *writer* side
  (issues #5, #6).

- **`build.impl` assignment** (knobs.py:326,
  `runtime/build.py:62` reader using `:=`). The reader uses walrus
  assignment: `if impl := knobs.build.impl:` — single read into a
  local. No TOCTOU. If this pattern were to change to a double-read
  `if knobs.build.impl is not None: knobs.build.impl(...)`, it would
  fall under issue #3.
