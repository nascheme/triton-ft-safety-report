# Threading Glossary for Free-Threaded Python Audits

A terminology guide for people reviewing data-race and timing issues in
free-threaded (nogil) Python code.  It primarily targets Python and
Python-adjacent runtime code; terms that are mainly relevant to native
extensions are marked explicitly.  Terms are organized from foundational
concepts through specific bug patterns to fix patterns.

---

## Foundational concepts

**Data race** — Two threads access the same mutable state concurrently, at
least one is a write, and there is no synchronization establishing a
happens-before relationship between them.

**Race condition** — Non-deterministic behavior caused by the relative timing
of concurrent operations.  A data race is one cause; logic-level races (TOCTOU)
are another.

**Happens-before relationship** — A guarantee that one operation's effects
are visible to a subsequent operation.  Without it, a reader may observe
stale or partial state.  In Python audits, the practical symptom is usually an
operation observing inconsistent shared state across multiple steps, not
hand-written atomic memory ordering.

**Critical section** — A code region where only one thread executes at a time,
typically enforced by a lock.

**Atomicity violation** — A sequence of operations that must appear indivisible
but isn't.  An observer thread can see the intermediate state between steps.

**Ordering violation** — Correctness depends on updates becoming visible in a
particular order, but concurrent execution exposes them in a different one.
Example: code publishes a "ready" flag before publishing the data that the
flag is supposed to guard.

**GIL as implicit lock** — Code that relied on the GIL to serialize access to
shared state.  Under free-threading the GIL no longer provides this
serialization, so previously-safe code becomes racy.  This is the root cause
behind most free-threading bugs: not new concurrency, but removed
serialization.

**Thread state attach / detach** *(native/C-extension code)* — In
free-threaded Python, `PyGILState_Ensure` / `PyGILState_Release` (and
pybind11's `py::gil_scoped_acquire` / `py::gil_scoped_release`) attach or
detach the calling thread's Python thread state.  This is required before
calling Python C APIs, but it **does not provide mutual exclusion** — multiple
threads can be attached simultaneously.  Prefer the terms "attach" and "detach"
over "acquire the GIL" and "release the GIL" when describing free-threaded
behavior, because the latter implies locking semantics that no longer exist.
Code that relies on `PyGILState_Ensure` as a lock to protect shared native
state is broken under free-threading and needs explicit synchronization.

---

## Classic race patterns

**TOCTOU (Time-of-Check-Time-of-Use)** — A check (`if key not in d`) becomes
stale before the action (`d[key] = value`) executes, because another thread
intervened.  The most common pattern in practice.

**Check-then-act** — The general form of TOCTOU: read a condition, then act on
it, without holding the condition stable.  Examples: `if k in d: del d[k]`,
`if obj is None: obj = create()`.

**Read-modify-write** — A non-atomic triplet: read a value, compute a new
value, write it back.  Example: `d[k] = d[k] + 1`.  Another thread's write
between the read and the write is silently lost.

**Lost update** — Two concurrent writers where one's result is silently
discarded.  Last-writer-wins without the loser being aware.

**Stale read** — Reading an outdated value because no synchronization forces
visibility of a more recent write.  Ranges from benign (debug flag) to severe
(wrong driver handle).

**Stale iterator / borrowed reference** *(mostly native/C-extension code)* —
An iterator, view, or borrowed native reference is obtained while shared state
is momentarily stable, then used after that stability is gone.  Concurrent
mutation can invalidate the traversal or the native reference.  At pure
Python level this is usually an iterator/view-consistency issue, not a
literal dangling pointer.

**Lock order inversion (ABBA deadlock)** — Thread A holds lock A and waits for
lock B; thread B holds lock B and waits for lock A.  Both block forever.
Relevant when callbacks acquire locks in the reverse order of the normal call
path.

---

## Lazy initialization races

**Lazy initialization race** — Multiple threads simultaneously trigger
first-use initialization of a shared resource.  Both see "not yet initialized,"
both do the work, and the results collide.

**Publish partially initialized / publish-before-init** — A reference to an
object is stored into shared state before the object's construction is
complete.  Another thread follows the reference and observes a half-built
object.

**Partially populated state** — Multiple attribute assignments in sequence are
not atomic as a group.  A reader between assignments sees some attributes set
and others still `None`.

**Non-atomic multi-target assignment** — A statement like
`obj.a, obj.b, obj.c = values` performs multiple writes in sequence.  Each
individual write may be safe under free-threading, but the group is not atomic
— a reader can observe a mix of old and new values.

**Double-checked locking** — A pattern to reduce lock contention on lazy init:
check without the lock, acquire the lock, check again.  Correct in principle
but easy to get wrong if the "publish" step isn't ordered properly.

**Singleton violation** — `__new__` returns a cached instance but `__init__`
still re-runs on every call, causing repeated initialization on an
already-shared object.

**Discarded initialization result** — When multiple threads race to lazily
initialize a shared slot, each may create an object but only one wins the
store.  In Python this is usually wasted work plus a discarded temporary
result; in native code it can become a true refcount or ownership leak if the
losing objects are not cleaned up correctly.

---

## Container and iteration races

**Unsafe iteration under concurrent mutation** — Iterating a list, dict, or
set while another thread mutates it.  Free-threaded CPython won't crash, but
the observed element sequence is undefined: elements may be skipped, revisited,
or partially observed.  Probably the most commonly missed bug category.

**Defaultdict auto-vivification** — `d[missing_key]` on a `defaultdict` is a
compound operation (check, call factory, insert) that is not atomic.  Two
threads hitting the same missing key can race.

**Live module dict iteration** — Iterating `fn.__globals__` (the module dict)
while another thread does module-level assignment.  Same undefined-sequence
problem as any dict iteration under mutation.

---

## Cache and memoization races

**Duplicate computation / duplicate work** — An expensive operation
(compilation, autotuning) runs multiple times concurrently when it should run
once.  Results from wasted resources to inconsistent cached results.

**Cache-key divergence** — A cache key is computed from state that changes
between reads (e.g., a hook reference is swapped mid-computation), so the key
doesn't match the value that gets stored.

**Future memoization race** — Two threads both miss a memoization check and
independently submit async work for the same key.  Leads to duplicate
submissions and duplicate finalization callbacks.

**Placeholder vs. finalize ordering** — An async placeholder (e.g., a future
object) is written into a cache slot *after* the real result has already been
finalized there, reverting the entry to an incomplete state.

---

## Hook and callback races

**Hook chain iteration race** — Iterating a list of callbacks while another
thread appends to or removes from the list.  Hooks may be skipped,
double-invoked, or invoked in an undefined order.

**Duplicate hook invocation** — A user-supplied callback fires multiple times
for a single event because finalization itself races.  Breaks once-per-event
contracts (counters, idempotency assumptions).

**Weakref callback race** — A weakref callback fires while related shared
state is being concurrently mutated or torn down by another thread.  The
callback timing is non-deterministic under free-threading, so the callback may
observe partially torn-down state or violate once-only assumptions.

**Register-then-teardown** — Registering a callback (e.g., a GC callback,
finalizer, or native hook) whose later execution assumes an object or runtime
state is still valid, then tearing that state down while the callback can
still fire.  In Python this usually means the callback sees already-cleared or
partially finalized state; in native code it can become a true lifetime bug.

---

## Configuration and selection races

**Configuration mutation race** — Configuration (knobs, hooks, driver/backend
selection) is mutated while another thread is mid-operation.  Different steps
of a single operation see different configuration state.

**Driver/backend selection race** — Multiple reads of a shared driver or
backend reference within the same function see different values because a
setter ran between reads.  Parts of one operation use mismatched instances.

**Mixed-version instance attributes** — Parallel initialization writes
attributes from different versions/sources.  The resulting instance contains a
mix of attributes from version A and version B.

**Config-as-global anti-pattern** — Using a module-level global variable to
pass configuration that should be per-thread or per-context.  Under
free-threading, one thread's context manager mutating the global corrupts
another thread's view.

**Fast-path / slow-path synchronization mismatch** — The common (fast) path
skips locking for performance, but a rare (slow) path acquires a lock.  The
fast path races with the slow path because they don't coordinate.

---

## Lifecycle and teardown races

**Child thread teardown race** — The main thread destroys shared state (module
unload, interpreter shutdown) while child threads still hold live references
and are actively using it.

---

## Consequence categories

**Benign race** — A race that at worst causes redundant computation with
identical results.  Not worth fixing unless it's expensive.

**Silent correctness issue** — No crash, no exception, but wrong results or
skipped checks due to stale/inconsistent reads.

**Spurious error** — An exception raised not because of a real error but
because of timing (e.g., calling `None` because a hook reference was swapped
mid-read).

**Duplicate benchmarking interference** — Concurrent autotuning runs interfere
with each other's timing measurements, producing unreliable performance data.

---

## Fix patterns

**Copy-on-write (snapshot publication)** — The writer copies a mutable
container, modifies the copy, then rebinds the shared reference atomically.
Readers always see a stable snapshot.  The most commonly suggested fix for
iteration races.

**Per-key lock** — Fine-grained locking where each cache key has its own lock.
Reduces contention vs. a single global lock.

**Per-device lock** — Intermediate granularity: one lock per device, protecting
all cache entries for that device.

**Per-thread or per-context isolation** — State is moved out of shared global
storage and made local to a thread (`threading.local()`) or execution context
(`contextvars.ContextVar`).  This can eliminate sharing entirely, but
`ContextVar` is context-local rather than simply thread-local.

**Deferred cleanup** — Instead of mutating shared state inside a callback or
hot path (where locking is dangerous or expensive), queue the mutation for a
later phase when synchronization is safe.

**Poisoning** — A design pattern where failed initialization marks state as
permanently broken so future callers re-raise cheaply instead of retrying.  A
race can violate the poisoning contract by letting unpoisoned partial state be
observed.
