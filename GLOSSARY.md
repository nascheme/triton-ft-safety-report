# Threading Glossary for Free-Threaded Python Audits

A terminology guide for people reviewing data-race and timing issues in
free-threaded (nogil) Python code.  Terms are organized from foundational
concepts through specific bug patterns to fix patterns.

---

## Foundational concepts

**Data race** — Two threads access the same mutable state concurrently, at
least one is a write, and there is no synchronization establishing a
happens-before relationship between them.

**Race condition** — Non-deterministic behavior caused by the relative timing
of concurrent operations.  A data race is one cause; logic-level races (TOCTOU)
are another.

**Happens-before relationship** — A memory-ordering guarantee that one
operation's effects are visible to a subsequent operation.  Without it, a
reader may observe stale or partial state.

**Critical section** — A code region where only one thread executes at a time,
typically enforced by a lock.

**Atomicity violation** — A sequence of operations that must appear indivisible
but isn't.  An observer thread can see the intermediate state between steps.

**Ordering violation** — Writes that land in an unexpected order, so a later
write overwrites an earlier, more-correct value.

**GIL as implicit lock** — Code that relied on the GIL to serialize access to
shared state.  Under free-threading the GIL no longer provides this
serialization, so previously-safe code becomes racy.  This is the root cause
behind most free-threading bugs: not new concurrency, but removed
serialization.

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

**Stale reference** — A reference or iterator returned from a locked section
remains in use after the lock is released.  Concurrent modification can
invalidate the referent, making the reference dangle.  Distinct from "stale
read" — here the *reference itself* becomes invalid, not just the value that
was read.

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

**Non-atomic multi-target tuple unpack** — Python's `a, b, c = tuple` compiles
to sequential `STORE_ATTR` opcodes.  Each store is individually safe under
free-threading, but the group is not atomic — a reader can observe a mix of old
and new values.

**Double-checked locking** — A pattern to reduce lock contention on lazy init:
check without the lock, acquire the lock, check again.  Correct in principle
but easy to get wrong if the "publish" step isn't ordered properly.

**Singleton violation** — `__new__` returns a cached instance but `__init__`
still re-runs on every call, causing repeated initialization on an
already-shared object.

**Reference leak from concurrent initialization** — When multiple threads race
to lazily initialize a shared slot, each creates an object but only one wins
the store.  The losing threads' objects are silently leaked if not properly
cleaned up.

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

**Weakref callback race** — A weak reference's destructor callback fires and
tries to access or modify an object that is being concurrently destroyed or
mutated by another thread.  The callback timing is non-deterministic under
free-threading.

**Register-then-destroy** — Registering a callback (e.g., a GC callback or
finalizer) that holds a reference to an object, then destroying the object
while the callback can still fire.  The callback dereferences a dangling
reference.

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

**Per-thread isolation / thread-local storage** — Each thread maintains its own
copy of state (via `threading.local()` or `contextvars.ContextVar`),
eliminating sharing entirely.  The strongest fix but not always applicable.

**Deferred cleanup** — Instead of mutating shared state inside a callback or
hot path (where locking is dangerous or expensive), queue the mutation for a
later phase when synchronization is safe.

**Poisoning** — A design pattern where failed initialization marks state as
permanently broken so future callers re-raise cheaply instead of retrying.  A
race can violate the poisoning contract by letting unpoisoned partial state be
observed.
