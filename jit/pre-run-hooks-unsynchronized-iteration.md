# `JITFunction.pre_run_hooks` — unsynchronized iteration during concurrent mutation

- **File:** `python/triton/runtime/jit.py`
- **Severity:** Significant
- **Tier:** 3 (configuration mutation concurrent with kernel launch)

## Shared state

`JITFunction.pre_run_hooks: list` — a plain Python list of user-supplied
callables. Initialized in `__init__` at jit.py:804:

```python
self.pre_run_hooks = []
```

The list instance is never rebound after construction; only its contents
are mutated.

## Writer

`JITFunction.add_pre_run_hook` (jit.py:663-669):

```python
def add_pre_run_hook(self, hook):
    '''
    Add a hook that will be executed prior to the execution of run
    function with args and kwargs passed into the kernel
    '''
    assert callable(hook)
    self.pre_run_hooks.append(hook)
```

No lock. `list.append` is individually atomic in free-threaded CPython.

## Reader

`JITFunction.run` (jit.py:716-718), on every kernel launch:

```python
# Execute pre run hooks with args and kwargs
for hook in self.pre_run_hooks:
    hook(*args, **kwargs)
```

No lock. Plain `for`-loop iteration.

## Race scenario

Thread A is in the middle of executing

```python
for hook in self.pre_run_hooks:
    hook(*args, **kwargs)
```

while Thread B calls `self.add_pre_run_hook(new_hook)`.

Per the Python thread safety documentation
(https://docs.python.org/dev/library/threadsafety.html#thread-safety-list):

> "Operations that involve multiple accesses, as well as iteration, are
> never atomic. [...] NOT thread-safe: iteration while modifying"
>
> "Consider external synchronization when sharing list instances across
> threads."

In free-threaded CPython, `list.append` may reallocate the backing array
when the list grows past its current capacity. The iterator object holds
an index into the list and reads `lst[i]` on each `__next__`. A concurrent
`append` that triggers reallocation races with the iterator's indexed
reads, and the observed sequence of elements is undefined:

- A previously registered hook may be **skipped**.
- A previously registered hook may be **visited twice**.
- The newly appended hook may or may not fire on the in-flight call.
- The interpreter does not crash — list operations are internally atomic
  enough to prevent segfaults or dangling reads — but the list content
  observed during iteration is not guaranteed to match any serialized
  order of the append.

## Consequence

`pre_run_hooks` is a user-facing extension point for profiling,
argument validation, logging, and similar instrumentation. The concrete
correctness failures are:

1. **Silently skipping a registered hook** on a concurrent `run()` call.
   If the hook performs validation (e.g. checking tensor shapes,
   asserting device placement, recording telemetry), the check is lost
   for that launch with no diagnostic.

2. **Double-invoking a hook** on a single `run()` call. Most hooks are
   idempotent, but some are not (counters, once-per-launch logging,
   state machines).

3. **Visiting hooks in an unexpected order** — not an issue today since
   hook ordering is already unspecified, but worth noting.

No crash, no interpreter corruption, no wrong kernel launched. The
launched kernel itself is unaffected. The failure mode is strictly in
the hook invocation sequence.

The window is wide: any concurrent `add_pre_run_hook` + `run()` pair on
the same `JITFunction` hits it. This is Tier 3 (concurrent configuration
mutation), which Triton considers a desirable support target.

## Suggested fix

**Copy-on-write.** Adding a hook is rare; iterating hooks happens on every
kernel launch. The reader path should pay nothing, and the writer should
do the work:

```python
# In JITFunction.__init__
self._pre_run_hooks_lock = threading.Lock()
self.pre_run_hooks = []   # treated as immutable after publish

# In add_pre_run_hook — copy, append, rebind
def add_pre_run_hook(self, hook):
    assert callable(hook)
    with self._pre_run_hooks_lock:
        hooks = list(self.pre_run_hooks)
        hooks.append(hook)
        self.pre_run_hooks = hooks

# In run — unchanged; single attribute read, iterate the snapshot
for hook in self.pre_run_hooks:
    hook(*args, **kwargs)
```

Why this works:

1. The list published to `self.pre_run_hooks` is never mutated again —
   every `add_pre_run_hook` builds a fresh list and rebinds. A reader
   that loads `self.pre_run_hooks` gets a stable snapshot and can iterate
   it freely, even if a concurrent `add_pre_run_hook` is in flight.
2. Attribute rebinding on an instance (`self.pre_run_hooks = hooks`) is
   itself a single `dict.__setitem__` on the instance `__dict__`, which
   is safe in free-threaded CPython. The reader either sees the old
   list or the new list, never a torn reference.
3. The writer lock (`_pre_run_hooks_lock`) is only needed to prevent
   **lost updates** between concurrent writers: without it, two threads
   could both load the same old list, both append their own hook, and
   whichever rebinds second wins — losing the other hook. Since
   `add_pre_run_hook` is rare, the lock is essentially uncontended.
4. No lock is held across user hook code. Hook callbacks can be slow
   and can re-enter Triton; holding a lock across them would be a
   deadlock risk.
5. The reader path (`run()` at line 717) requires **no change** — the
   existing `for hook in self.pre_run_hooks:` is already correct once
   the list is treated as immutable after publish. That makes this fix
   zero-cost on the hot path.

The pattern relies on "the published list is never mutated again."
Anyone extending `pre_run_hooks` must use the copy-on-write writer, not
`self.pre_run_hooks.append(...)` directly. A short comment on the
attribute is worth adding to document the invariant.
