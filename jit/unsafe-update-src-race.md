# `JITCallable._unsafe_update_src` — unsynchronized hash invalidation

- **File:** `python/triton/runtime/jit.py`
- **Severity:** Significant
- **Tier:** 3 (configuration mutation while another thread compiles)

## Shared state

`JITCallable._src` and `JITCallable.hash`. These two attributes must stay in
sync: `self.hash`, once populated, is meant to be the hash of whatever is
currently in `self._src`. Every other path that mutates them goes through
`cache_key` and takes `self._hash_lock` (an `RLock`) to enforce that
invariant.

`_unsafe_update_src` is the one exception.

## Writer

`JITCallable._unsafe_update_src` (lines 543–551):

```python
def _unsafe_update_src(self, new_src):
    """
    The only method allowed to modify src.
    ...
    Note that it is the callers responsibility to make sure any triton
    functions that call this function have the `.hash` value reset to None.
    """
    self.hash = None
    self._src = new_src
```

No lock is acquired. Two writes happen independently: `self.hash = None`
first, then `self._src = new_src`.

The lock-respecting writer is `JITCallable.cache_key` (lines 499–520), which
holds `self._hash_lock` while it reads `self.src` / calls `self.parse()`,
computes the hash, and publishes it:

```python
@property
def cache_key(self) -> str:
    with self._hash_lock:
        if self.hash is not None:
            return self.hash
        self.hash = f"recursion:{self._fn_name}"
        ...
        dependencies_finder.visit(self.parse())  # reads self._src
        self.hash = dependencies_finder.ret + str(self.starting_line_number)
        self.used_global_vals = dict(sorted(...))
        ...
        self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
    return self.hash
```

## Reader

`cache_key` (same method) is the canonical reader of both `self.hash` and
`self._src`. It is reached transitively on the compile hot path — e.g. from
`ASTSource` hashing in `compiler/compiler.py`, ultimately driven by
`_do_compile` at jit.py:884 — and from `compute_cache_key` at jit.py:590
whenever a JIT-callable appears in a specialization. `JITCallable.__hash__`
(line 522) also reads it.

## Race scenario

Both scenarios below assume Thread A is driving a kernel compile and Thread B
is the caller of `_unsafe_update_src` (e.g. a user monkey-patching the source
of an already-live `JITFunction`, or the `_unsafe_update_src` call at
`triton_kernels/specialize.py:64` if the caller ever shares the target
function before replacement finishes).

### Scenario 1 — lost invalidation (primary bug)

1. Thread A calls `cache_key`, acquires `_hash_lock`, observes
   `self.hash is None`, reads `src=self.src`, walks the AST of **old** src,
   and starts computing the hash.
2. Thread B calls `_unsafe_update_src(new_src)`. Because it does not take
   `_hash_lock`, it does not wait. It writes `self.hash = None` (Thread A's
   in-progress placeholder, about to be overwritten anyway) and then
   `self._src = new_src`.
3. Thread A finishes the critical section and writes
   `self.hash = <sha256 of old-src dependencies>` at line 519. Its write
   "wins" — the `hash = None` that Thread B just performed is clobbered.
4. Any subsequent reader of `cache_key` sees `self.hash is not None` and
   returns the **old-src** hash, even though `self._src` is now `new_src`.

The documented invariant that `_unsafe_update_src` resets the hash is
silently violated.

### Scenario 2 — inconsistent reads of `self._src` inside one `cache_key` call

Attribute loads in free-threaded CPython are atomic, a single read of
`self._src` always returns either the full old reference or the full new
reference. The problem is that `cache_key` reads `self._src` **twice** and
those two reads are not tied together.

1. Thread A holds `_hash_lock` and at line 510 reads `src=self.src` — gets
   the **old** src. It constructs `dependencies_finder` with that old src
   (used later for error messages).
2. Thread B runs `_unsafe_update_src` and replaces `self._src` with
   `new_src`.
3. Thread A continues to `dependencies_finder.visit(self.parse())` at line
   511. `parse()` re-reads `self._src` and parses the **new** src. The
   finder now walks the new-src AST while holding the old src string for
   diagnostics.
4. Thread A publishes a hash derived from the new-src AST. Whether that
   hash ends up matching any future observed `self._src` depends on how
   this interleaves with Scenario 1.

The net effect is that there is no well-defined relationship between the
cached `self.hash` and the value of `self._src` visible to the next reader.

## Consequence

`cache_key` is the identity Triton uses to look up compiled kernels
(`compute_cache_key` at line 574 calls `obj.cache_key` for every nested
`JITCallable` in a specialization). A stale `self.hash` after an
`_unsafe_update_src` means:

- Compiles for the new src reuse the cache entry of the old src. Callers
  launch a stale compiled kernel even though the source was updated.
- Conversely, if Scenario 2 lands first, the new-src hash can leak into an
  entry keyed against the old src, making a later revert miss the cache.

This is not a crash and not container corruption — both `self.hash` and
`self._src` are single-attribute writes on a normal Python object, and
free-threaded CPython serializes each individual attribute store. The bug
is purely about **ordering**: the two stores in `_unsafe_update_src` and the
stores inside `cache_key`'s critical section are not ordered relative to
each other, so the invariant "hash reflects current src" can be broken.

Tier 3: this requires a mutation (`_unsafe_update_src`) to run concurrently
with a compile path. In production Triton the main in-tree caller
(`triton_kernels/specialize.py:64`) runs on a freshly-constructed
`JITFunction` before it is shared, so that call site is safe today. The
hazard is for user code that calls `_unsafe_update_src` on a `JITFunction`
that other threads may still hold references to, plus the existing in-tree
unit tests at `python/test/unit/runtime/test_cache.py:78,80,120–126,148,153`
and `python/test/unit/language/test_core.py:110` that mutate live kernels.

The method name `_unsafe_update_src` and its docstring warn that the caller
must clear `.hash` manually, but they describe an intra-thread obligation.
They do not cover the free-threading case where another thread is
simultaneously inside `cache_key`.

## Suggested fix

Acquire `self._hash_lock` inside `_unsafe_update_src` so that the src
replacement and the hash invalidation happen inside the same critical
section that `cache_key` uses. This ensures:

- No `cache_key` computation can be in flight while `_src` is swapped.
- The `self.hash = None` write cannot be clobbered by a concurrent in-flight
  `cache_key` finalization, because no such finalization can overlap.

```python
def _unsafe_update_src(self, new_src):
    with self._hash_lock:
        self.hash = None
        self._src = new_src
```

`_hash_lock` is an `RLock`, so this is safe even if a caller somehow reaches
`_unsafe_update_src` from inside a `cache_key` re-entrancy chain (not
expected, but the reentrant lock removes any risk). The cost is one
uncontended lock acquire per src update, which is negligible compared to
everything else `_unsafe_update_src` triggers downstream.
