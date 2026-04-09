# `JITCallable.used_global_vals` — unsynchronized read skips the global-changed safety check

- **File:** `python/triton/runtime/jit.py`
- **Severity:** Significant
- **Tier:** 2 (multiple threads calling the same JIT kernel)

## Shared state

`JITCallable.used_global_vals` — a `Dict[Tuple[str, int], Tuple[Any, Dict]]`
attached to the `JITFunction` instance. It records the global variables a
kernel captured and their values at the time of first compilation, so that
`run()` can detect mutation of those globals between calls.

Initialized to `{}` in `__init__` (line 483) and then rebound to a populated
dict inside `cache_key` (line 513).

## Writer

`JITCallable.cache_key` (lines 499–520):

```python
@property
def cache_key(self) -> str:
    with self._hash_lock:
        if self.hash is not None:
            return self.hash
        ...
        self.hash = dependencies_finder.ret + str(self.starting_line_number)
        self.used_global_vals = dict(sorted(dependencies_finder.used_global_vals.items()))
        ...
```

Writes happen **exactly once** per JITCallable (guarded by `self.hash is not
None`) and are serialized by `self._hash_lock` (an `RLock`).

`cache_key` is invoked during compilation — e.g. from
`compiler/compiler.py:75` when `ASTSource` computes its hash — so the write
happens somewhere inside `self.compile(...)` in `_do_compile` (line 884),
*before* the kernel is published to `kernel_cache[key]` at line 885.

## Reader

`JITFunction.run()` (lines 743–748), with **no lock held**:

```python
# Check that used global values have not changed.
not_present = object()
for (name, _), (val, globals_dict) in self.used_global_vals.items():
    if (newVal := globals_dict.get(name, not_present)) != val:
        raise RuntimeError(
            f"Global variable {name} has changed since we compiled this kernel, "
            f"from {val} to {newVal}")
```

This loop runs on every `run()` call, including cache-hit paths that never
touched `cache_key` or `_hash_lock`.

## Race scenario

Two threads call `run()` on the same `JITFunction` for the first time with
the same specialization:

1. **Thread A** enters `_do_compile`, calls `self.compile(src, ...)` at line
   884. Inside compile, `ASTSource` hashing reads `self.fn.cache_key`,
   which acquires `_hash_lock`, walks the AST, and sets
   `self.used_global_vals = dict(sorted(...))` at line 513. The lock is
   released. Compile finishes. Thread A then publishes the kernel at line
   885: `kernel_cache[key] = kernel`.

2. **Thread B** runs `run()` concurrently. At line 732 it calls
   `kernel_cache.get(key, None)` — free-threaded CPython serializes this
   read against the dict's own internal critical section, and Thread B
   observes Thread A's published kernel. Thread B skips `_do_compile` and
   falls through to line 745: `for ... in self.used_global_vals.items(): ...`.

3. Python's free-threading memory model gives each mutable object its own
   critical section. The `kernel_cache` dict write and the
   `self.used_global_vals` attribute assignment happen under unrelated
   critical sections — there is no common lock tying them together, and
   no happens-before relationship between "Thread A wrote
   `used_global_vals`" and "Thread B read the kernel from `kernel_cache`."

4. Thread B can therefore legally read `self.used_global_vals` and still
   observe the initial empty dict from `__init__` (line 483). It iterates
   zero entries and skips the safety check entirely.

## Consequence

Silent correctness issue: the "global variable X has changed since we
compiled this kernel" guard (lines 743–748) is skipped for Thread B's call.
If a captured global actually changed between the two calls, Thread B will
run the stale kernel without warning.

This is not a crash. `used_global_vals` is rebound (never mutated in place),
so the iteration itself is on a stable dict snapshot — there is no
container corruption or "changed size during iteration" error.

The race window is narrow: only the **first** concurrent call burst matters,
because after `cache_key` runs once, `used_global_vals` stays populated for
the life of the `JITFunction` (absent `_unsafe_update_src`, see issue #11).
But for Tier 2 (multiple threads calling the same kernel the first time),
the window is real and the consequence — silently bypassing a user-facing
correctness check — is worth fixing.

## Suggested fix

The check at lines 743–748 needs to observe `used_global_vals` after a
synchronizing operation that pairs with the writer's lock release. Options:

1. **Acquire `_hash_lock` briefly before reading.** This is the simplest
   correct fix. The cost is one uncontended lock acquire per `run()` call,
   which is cheap compared to the rest of the launch path. Since the
   writer only runs once, the lock is almost always uncontended.

2. **Populate `used_global_vals` eagerly in `_do_compile` before publishing
   to `kernel_cache`.** Ensure the write-publish ordering uses the same
   lock that readers use — for example, by folding the cache-hit check and
   the globals-changed check under the same per-device lock that already
   needs to exist to fix the `kernel_cache` / `kernel_key_cache` TOCTOU
   (issue #4).

3. **Call `self.cache_key` (an RLock-protected property) once at the top
   of `run()`.** This is a no-op on the fast path after the first call,
   and guarantees the read observes a fully populated `used_global_vals`.

Option 3 is the least invasive change.

Example fix:

```python
    def run(self, *args, grid, warmup, **kwargs):
        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

        # Ensure used_global_vals is populated and its write is synchronized
        # with this reader via _hash_lock. After the first call this is a
        # cheap, uncontended RLock acquire that returns immediately.
        _ = self.cache_key

        # parse options
        device = driver.active.get_current_device()
        ...
```
