# How to Audit Triton for Free-Threading (nogil) Thread-Safety Issues

This directory contains reports for auditing Triton's Python-facing runtime for
issues related to Python 3.14t free-threading.

With the GIL removed, Python code, pybind11 bindings, and native helper code
that previously relied on implicit serialization can now have real data races.

Triton source is symlinked at `./triton`.

Primary scope:

- `triton/python/triton/` — Python runtime, JIT dispatch, compiler orchestration,
  caches, knobs, backend/driver selection, autotuning
- `triton/python/src/` — C++/pybind11 bindings used by the Python layer

The core MLIR / LLVM compiler internals are out of scope except where they:

- touch Python-visible state,
- hold shared mutable native state,
- or run in GIL-released regions.

See [OVERVIEW.md](OVERVIEW.md) for the high-level architecture.
See [AUDIT_PLAN.md](AUDIT_PLAN.md) for scope and recommended file ordering.
See [GLOSSARY.md](GLOSSARY.md) for a description of preferred terms.

## What to look for

Focus on **shared mutable state and data flow**, not surface-level pattern
matching.

The key question is not "does this file mention threads?" The key question is:

- what state is shared,
- who writes it,
- who reads it,
- and can those accesses happen concurrently under free-threading?

### High-value patterns (these cause real bugs)

1. **Lazy native initialization of shared C++ state.**
   `python/src/specialize.cc` is the highest-priority example. It has lazy
   process-global initialization (`init_globals()`) guarded only by a plain
   boolean (`init_called`), plus static native caches such as
   `dtype_ptr2str`, `dtype2str`, and `type_handler_cache`. Two threads reaching
   specialization concurrently can race on initialization and on later cache
   mutation.

2. **Shared mutable caches attached to reusable Python wrapper objects.**
   `JITFunction` and `Autotuner` objects are intended to be reused across many
   calls, including calls from different threads. Mutable state hanging off
   these objects is therefore shared state. High-priority examples include:
   - `JITFunction.device_caches`
   - key/result caches used during specialization and compilation
   - `Autotuner.cache`
   - async compile bookkeeping that memoizes or finalizes work for shared keys

3. **Global registries and lazy singletons on the hot path.**
   Module-level registries and singleton objects are often safe under the GIL
   by accident and unsafe under free-threading. Important Triton examples
   include:
   - `_triton_jit_function_registry`
   - `runtime.driver.driver`
   - `backends.backends`
   - process-global `knobs.*` objects and their hook/callback state

4. **Lazy first-use state on shared Python objects.**
   `@cached_property`, ad hoc memoization, and first-use writes into object
   `__dict__` are worth attention when the object instance is shared. A good
   example is `KernelParam` in `runtime/jit.py`.

5. **Shared caches whose exact runtime guarantees are unclear.**
   Triton uses `@lru_cache` and other memoization helpers in a few places.
   Treat them as shared mutable state worth reviewing. If the exact
   free-threading behavior of a helper is unclear, leave a note for follow-up
   rather than guessing.

6. **Code running with detached thread state (GIL-released native regions).**
   `python/src/llvm.cc` and `python/src/ir.cc` contain explicit thread-state
   detach points (`py::gil_scoped_release` / `Py_BEGIN_ALLOW_THREADS`). Any
   Python C API use inside those regions is unsafe. Any native shared mutable
   state touched there is concurrently accessible.

7. **Compiled-kernel lazy finalization / lazy runtime handle init.**
   `CompiledKernel` is created by the compiler path but lazily initializes some
   runtime-facing handles on first real use. Two threads reaching that path on
   the same object are a classic free-threading audit target.

8. **Hooks/callback lists that can be mutated while the runtime is using them.**
   Triton exposes hook chains and callback slots through `knobs.py`. If one
   thread mutates these while another thread compiles, loads, or launches a
   kernel, that is shared mutable state on an active execution path.

### Pybind11 / Python-C-API specific patterns

9. **`py::call_guard<py::gil_scoped_release>()`.**
   When a binding is declared with `py::call_guard<py::gil_scoped_release>()`,
   the bound function runs with the thread state detached for that call.
   Audit the whole function body with that in mind: Python C API calls are
   forbidden, and any shared native state is concurrently accessible.

10. **Stack-based `py::gil_scoped_release allow_threads`.**
    Once constructed, the thread state is detached until the scope ends.
    Verify that no Python C API calls or unsafe shared-state access occur
    inside that region.

11. **Native containers storing `PyObject*` or Python-derived state.**
    Containers like `std::unordered_map<..., PyObject*>` are a double hazard:
    the container itself is not thread-safe, and the Python object lifetime /
    ownership assumptions may also be wrong. Verify both the container race and
    the refcount/lifetime story.

12. **Unsafe Python C APIs in thread-state-detached regions.**
    In detailed C++ passes, watch for APIs such as:
    - `PyObject_GetAttr`
    - `PyObject_Call*`
    - `PyDict_*`, `PyList_*`
    - `Py_BuildValue`, `PyArg_Parse*`
    - `Py_INCREF`, `Py_DECREF`
    - `PyErr_SetString`, `PyErr_Occurred`
    - container mutation APIs in general

    Also watch borrowed-reference APIs like `PyDict_GetItem` / `PyList_GetItem`
    whose results can become invalid if another thread mutates the underlying
    object.

### Low-value patterns (usually not worth reporting)

- **Write-once globals set during module init.**
  State created during `PYBIND11_MODULE(...)` initialization is generally
  protected by Python's import lock. Do not report it unless the code actually
  exposes concurrent access before initialization completes, or unless the
  state is later mutated.

- **C++11 magic statics.**
  Function-local `static` initialization is thread-safe by the C++ standard.
  Only flag these when the stored value itself is unsafe shared state
  (especially Python object pointers or Python-calling initializers).

- **Already-protected state.**
  Do not re-report synchronization that already exists unless you can show it is
  insufficient. For example:
  - `JITFunction._hash_lock` protects `cache_key` computation
  - `python/src/interpreter.cc` uses `std::mutex atomic_op_guard` for its
    fallback atomic operations

- **`ContextVar`-based state.**
  `_allocation._allocator` and `_async_compile.active_mode` are intentionally
  context-local. Verify that they stay context-local, but do not treat them as
  ordinary globals by default.

- **Benign debug/print flag staleness.**
  Many `knobs.*` flags control debugging, logging, or diagnostics. A stale read
  is usually not worth reporting unless it affects compilation correctness,
  cache behavior, backend/driver selection, or launch behavior.

- **Filesystem atomic replace by itself.**
  Triton's cache/build paths use filesystem operations such as atomic replace.
  That alone is not a bug. Focus on whether the surrounding shared Python/C++
  state coordinating the cache is raced.

- **The mere presence of `PyGILState_Ensure` / `PyGILState_Release`.**
  These attach/detach the calling thread's Python thread state and are required
  before calling Python C APIs, but under free-threading they do **not** provide
  mutual exclusion. The issue is when code relies on them as an implicit lock
  protecting native shared state. See GLOSSARY.md "Thread state attach / detach".

## Python built-in type thread safety

Free-threaded CPython protects most built-in container operations with
per-object critical sections, but the rules are subtle and not widely
known. **Before reporting a race on a `list`, `dict`, `set`, `bytearray`,
`memoryview`, `collections.*`, or `queue.*` object, read
[PYTHON_THREADSAFETY.md](PYTHON_THREADSAFETY.md)** — it summarizes what
is and isn't safe on free-threaded CPython, with pointers to
[https://docs.python.org/dev/library/threadsafety.html](https://docs.python.org/dev/library/threadsafety.html).

Quick rules of thumb:
- Single ops (`lst.append`, `d[k] = v`, `s.add`) on shared containers are
  safe — don't report them as corruption.
- Iteration under concurrent mutation is **not** safe, even for lists and
  dicts. The interpreter doesn't crash but the observed element sequence
  is undefined. This is the most commonly missed bug.
- Check-then-act (`if k in d: del d[k]`) and read-modify-write
  (`d[k] = d[k] + 1`) are never safe.
- `collections.defaultdict` auto-vivification (`d[missing]`) is a
  compound op and is a real race on shared defaultdicts.

## Support Model

The rank model lives in [`issues/README.md`](issues/README.md#rank-model). It
is the authoritative version; do not redefine rank meanings in component issue
files or in this document.

When auditing, decide whether the issue affects the current free-threading
support goal or should be deferred. The current goal includes one Triton thread
running while unrelated Python threads run, and multiple Triton threads
compiling or launching kernels concurrently. Triton maintainers already use
multi-threaded compilation in production under the GIL build, so concurrent
compilation is current-goal behavior.

Deferred issues include concurrent mutation of `knobs.*`, hook chains,
driver/backend selection, or `TRITON_INTERPRET=1` while other threads compile
or launch. These are tracked for completeness, but the underlying scenarios are
unsupported even under the GIL build today.

If an issue belongs to an out-of-scope category listed in
[`issues/README.md`](issues/README.md#rank-model), say so explicitly rather
than promoting it to a current-goal rank.

## How to audit a file

For each piece of mutable state you find:

1. **Who writes it?**
   Trace all writers. Pay special attention to lazy initialization, cache-fill
   paths, finalization callbacks, and configuration mutation.

2. **Who reads it?**
   Trace all readers. Is the reader on a hot dispatch/compile/load path? Is the
   same object reused across threads?

3. **Can the read and write happen concurrently?**
   Under free-threading, two Python threads can execute simultaneously. If a
   write happens on first use or during cache population, assume a second thread
   may arrive at the same time unless proven otherwise.

4. **What is the concrete consequence?**
   Construct a scenario: "Thread A is doing X in function F while Thread B is
   doing Y in function G; shared state Z is corrupted / observed half-initialized
   because ..." If you cannot describe the scenario, the issue is probably not
   yet well-founded.

5. **What breaks?**
   Classify the consequence:
   - crash / container corruption / use-after-free
   - duplicate compile or duplicate finalization
   - wrong cache hit/miss behavior
   - wrong backend/driver/config selection
   - benign duplicate computation or stale read

## Report format for component

Create a file `issues/<component>.md` file.  At the top of that file, create a
table:

  ```markdown
  ## Issues

  | # | Rank | Component | Issue |
  |---|------|-----------|-------|
  |   |      |           |       |
  ```

After the table, create another section "## Triage notes".

The triage notes section is the place for:
- extra context,
- alternative hypotheses,
- rank reasoning,
- examples / interleavings,
- why a suspected issue was rejected or downgraded,
- and any detail that would make the individual issue files too long.

For issues that are determined to be valid, create one individual markdown file
per issue as `issues/<component>/<issue-name>.md`.

Keep the individual issue files brief. Reviewers should be able to understand
the bug, impact, and fix quickly without reading a long narrative. Put any
supporting detail or extended discussion in `issues/<component>.md` under
"## Triage notes".


## Report format for individual issues

Individual issue files should be **brief** and should usually describe **one
issue only**. They are reviewed by expert Python programmers with limited time.

Prefer a short, flat format like this:

```markdown
# `<short issue title>`

- **Status:** Open
- **Rank:** Critical Blocker | Blocker | Deferred | Low | Rejected | Out-of-scope
- **Component:** `<file or subsystem>`

- **Shared state:** what shared mutable state is being raced on
- **Writer(s):** who writes it and when
- **Reader(s):** who reads it and when
- **Race scenario:** concise concrete interleaving: "Thread A does X while
  Thread B does Y -> consequence"
- **Suggested fix:** brief and concrete
```

Guidelines:
- Keep issue files as short as possible while still making the bug clear.
- Prefer one screen of text if possible.
- Do **not** repeat long architecture background, rank discussions, or
  alternative theories in the issue file unless they are necessary to establish
  the bug.
- Put extra commentary, nuance, edge cases, examples, and follow-up notes in
  `issues/<component>.md` under `## Triage notes`.
- If a component has multiple closely related races, use separate issue files
  when reviewers would likely want to discuss or fix them independently.
- If several smaller races are really consequences of one root design bug, keep
  the individual issue file focused on the root issue and put the decomposition
  into `issues/<component>.md`.

### Style note

Use the individual issue files for the minimum needed to answer:

1. What shared state is raced?
2. How can two threads reach it concurrently?
3. What breaks?
4. What is the fix direction?

If you find yourself adding long caveats, multiple alternative hypotheses, or
detailed rank justification, move that material into the component
`issues/<component>.md` triage notes instead.

## Rank Guide

- **Critical Blocker:** Current-goal issue with plausible crash, memory/native
  container corruption, borrowed-reference invalidation, wrong-kernel
  execution, or corrupted runtime/compiler state.

- **Blocker:** Current-goal issue that causes incorrect behavior, duplicate
  compile/finalization with observable side effects, incorrect cache behavior,
  wrong backend/driver selection, or unsafe shared mutable instance state, but
  with less catastrophic expected consequences than Critical Blocker.

- **Deferred:** Serious race shape that requires unsupported concurrent
  configuration mutation, hook mutation, driver/backend switching, debug-mode
  execution, or `TRITON_INTERPRET=1`. These are tracked, not current blockers.

- **Low:** Duplicate computation, low-impact first-use races, or stale reads
  where the consequence is limited.

- **Rejected / Out-of-scope:** Import-time one-shot state, `ContextVar` state,
  harmless debug-flag staleness, already-protected state, or vague suspicions
  without a concrete race scenario. Use Out-of-scope for real concerns that are
  not free-threading-specific Triton asks.

## Generating patch files

Use `git` to create patch files.  First, checkout the source in a clean state.
Apply your fixes and then commit.  Use `git format-patch` to create the patch
file. Add `X-Rank` and `X-Issue-Id` headers (`FT<nnn>`) to the patch file.
Reset the `triton` repo to it's original state.

## Fix hazard: do not break pickling when adding a lock

**Before adding a `threading.Lock`/`RLock` (or any other unpicklable object) as
an instance attribute, check whether that object is pickled.** `torch.compile`'s
Inductor caching serializes Triton runtime objects into a *persistent on-disk
cache*, so a lock that is correct under free-threading can still break
`torch.compile` (on **both** the GIL and free-threaded builds — this is not a
free-threading-only failure).

Known pickle paths (verify against the installed torch, these move around):

- `TritonBundler.put_static_autotuner` (`torch/_inductor/triton_bundler.py`)
  does `copy.deepcopy(caching_autotuner)` then pickles it. The
  `CachingAutotuner` reaches Triton's **`CompiledKernel`** via
  `compile_results[].kernel`, and the **`JITFunction`** via `.fn`.
- Combo-kernel autotuning (`generate_and_run_autotune_block`) pickles through
  the same kinds of objects.

Symptom: `cannot pickle '_thread.lock' object`, raised at cache-write time
(e.g. during the first profile run), *not* from a thread race.

torch only hand-strips the upstream `JITFunction._hash_lock` (in
`CachingAutotuner.prepare_for_pickle` / `restore_after_unpickle`,
`torch/_inductor/runtime/triton_heuristics.py`). It has no knowledge of any new
lock we add.

**Required fix when you add an instance lock to a pickled object:** give that
object `__getstate__`/`__setstate__` that drop the lock on serialize and
recreate a fresh one on restore — the same idiom torch's own `CachingAutotuner`
uses for its `lock`. Example (`CompiledKernel`, FT010):

```python
def __getstate__(self):
    state = self.__dict__.copy()
    state["_init_lock"] = None   # threading.Lock is not picklable
    return state

def __setstate__(self, state):
    self.__dict__.update(state)
    self._init_lock = threading.Lock()
```
