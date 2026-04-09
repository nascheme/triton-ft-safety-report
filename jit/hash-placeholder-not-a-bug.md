# `JITCallable.hash` placeholder visibility — not a bug

- **Status:** Investigated, not a bug
- **Severity:** None
- **Component:** `runtime/jit.py`
- **Source reports:** [README.md](README.md) (issue #10 and triage item #10)

## Original concern

The triage note raised a concern about lines 507–519 of `JITCallable.cache_key`:

```python
@property
def cache_key(self) -> str:
    with self._hash_lock:
        if self.hash is not None:
            return self.hash
        # Set a placeholder hash to break recursion in case the function
        # transitively calls itself. The full hash is set after.
        self.hash = f"recursion:{self._fn_name}"              # line 507
        nonlocals = inspect.getclosurevars(self.fn).nonlocals
        dependencies_finder = DependenciesFinder(...)
        dependencies_finder.visit(self.parse())
        self.hash = dependencies_finder.ret + str(...)        # line 512
        self.used_global_vals = dict(sorted(...))             # line 513
        ...
        self.hash = hashlib.sha256(...).hexdigest()           # line 519
    return self.hash
```

The worry: between lines 507 and 519 — still inside `_hash_lock` — `self.hash`
holds a placeholder string (`"recursion:<fn_name>"`). If *another thread*
were to read `self.hash` directly (not through `cache_key`), it would observe
the placeholder as a valid-looking hash and skip the early-return at line
503–504, returning the placeholder instead of the real hash.

`_hash_lock` is a `threading.RLock`. It serializes threads that go through
`cache_key`, but it does not protect direct reads of `self.hash` from other
code paths.

## Investigation

I searched the entire Triton Python tree for readers of `JITCallable.hash`
(the attribute, not other unrelated `.hash()` methods):

```
triton/python/triton/runtime/jit.py:472:        self.hash = None
triton/python/triton/runtime/jit.py:503:            if self.hash is not None:
triton/python/triton/runtime/jit.py:504:                return self.hash
triton/python/triton/runtime/jit.py:507:            self.hash = f"recursion:{self._fn_name}"
triton/python/triton/runtime/jit.py:512:            self.hash = dependencies_finder.ret + ...
triton/python/triton/runtime/jit.py:516:            self.hash += str([...])
triton/python/triton/runtime/jit.py:519:            self.hash = hashlib.sha256(...)
triton/python/triton/runtime/jit.py:520:        return self.hash
triton/python/triton/runtime/jit.py:550:        self.hash = None       # _unsafe_update_src
```

All eight references live inside `JITCallable.cache_key` or
`_unsafe_update_src`. Every read of `self.hash` is inside `cache_key` and
therefore under `_hash_lock`. No other code in Triton accesses the attribute
directly.

Other hash-related call sites go through the property, not the attribute:

- `JITCallable.__hash__` (line 522–523): `return hash(self.cache_key)` —
  goes through the property.
- `compiler/compiler.py:75`: `f"{self.fn.cache_key}-..."` — property.
- `compiler/compiler.py:421`: `self.hash = hash` — this is
  `CompiledKernel.hash`, a different object.
- `runtime/cache.py:316` and similar: `src.hash()` — method call on
  `ASTSource`, not `JITCallable.hash`.
- `runtime/autotuner.py:186`: `fn.cache_key` — property.

The placeholder at line 507 is purely an internal recursion-breaking device.
It is set, observed only by the same thread re-entering `cache_key`
(legitimate — that is why `_hash_lock` is an `RLock`), and then overwritten
before the lock is released. It never escapes the locked region because
nothing outside the locked region reads the attribute.

## Conclusion

**Not a bug.** The hypothesized reader that would observe the placeholder
does not exist. Every access to `self.hash` is inside the `_hash_lock`
critical section in `cache_key`.

If a future change introduced a direct read of `self.hash` (bypassing the
`cache_key` property), this concern would become real. A cheap defense would
be to rename the attribute to `_hash` and expose `cache_key` as the only
public reader, making the invariant structural rather than relying on
discipline. That is a style improvement, not a bug fix.

## Related issues

- The *related* concern about `self.used_global_vals` being visible in a
  partially populated state to `JITFunction.run()` line 745 is real and is
  tracked separately as issue #6
  ([used-global-vals-unsynchronized-read.md](used-global-vals-unsynchronized-read.md)).
  That issue is about `used_global_vals`, not `self.hash`, and has a
  concrete reader outside the lock.
- Issue #11 (`_unsafe_update_src` clearing `self.hash` without the lock) is
  a separate concern tracked in
  [unsafe-update-src-race.md](unsafe-update-src-race.md).
