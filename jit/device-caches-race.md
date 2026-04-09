# `JITFunction.device_caches` defaultdict auto-vivification race

- **Status:** Unreported
- **Severity:** SEVERE
- **Component:** runtime/jit.py
- **Tier:** 2
- **Source reports:** [README.md](README.md) (issue #2)

## Description

`JITFunction.device_caches` is a `defaultdict(self.create_binder)` initialized
at line 790. It is the central per-device dispatch table: every call to `run()`
(line 720), `preload()` (line 816), and `_do_compile()` (line 860) indexes into
it with a device key.

When a `defaultdict` encounters a missing key, `__missing__` calls the factory
function, inserts the result, and returns it. This check-miss-call-insert
sequence is **not atomic**; under free-threading, two threads that hit the same
missing key concurrently can both enter the factory, both create separate
values, and race on dict insertion.

There are two distinct hazards.

### Hazard 1: Duplicate factory execution and lost cache state

Two threads calling `run()` on the same `JITFunction` for the same device (both
first-time) will both find the key missing and both invoke `create_binder`.
Each call creates **fresh empty dicts** for `kernel_cache` and
`kernel_key_cache` (line 682: `return {}, {}, target, backend, binder`). The
second insertion overwrites the first, so whichever thread's entry survives in
the dict wins. The losing thread holds a reference to caches that are **not**
the ones stored in `device_caches` — any kernels it compiles and caches will be
silently lost. Subsequent calls will recompile those kernels.

Worse, if the losing thread is mid-way through `_do_compile` and writes to its
orphaned `kernel_cache` (line 882, 885), while a third thread reads from the
canonical `kernel_cache` obtained from a fresh `device_caches[device]` lookup,
the third thread will never see those compiled kernels and will trigger
redundant compilation.

### Hazard 2: `create_binder` mutates `self` as a side effect

`create_binder` (lines 671-682) writes three instance attributes on the
`JITFunction` object:

```python
self.CompiledKernel = CompiledKernel   # line 678
self.compile = compile                 # line 679
self.ASTSource = ASTSource             # line 680
```

These writes are **not guarded by any lock** and are **not specific to the
device key**. They are process-global imports assigned to instance attributes
for later use by `_do_compile` (line 865: `self.ASTSource(...)`, line 884:
`self.compile(...)`).

When two threads trigger `create_binder` for *different* devices concurrently,
both threads write these attributes in an interleaved order. This is a data
race on the `JITFunction` instance `__dict__`. In practice the values should be
identical (same imports), so the race is unlikely to produce wrong values — but
it is still undefined behavior under free-threading and could cause crashes if
the dict resizes concurrently.

When two threads trigger `create_binder` for the *same* device, the same writes
happen redundantly with the same risk.

## Shared state

- `self.device_caches`: `defaultdict` on a shared `JITFunction` instance
- `self.CompiledKernel`, `self.compile`, `self.ASTSource`: instance attributes
  written as a side effect of the factory

## Writer(s)

- `defaultdict.__missing__` via any `self.device_caches[device]` access with a
  new device key:
  - `run()` at line 720
  - `preload()` at line 816
  - `_do_compile()` at line 860

## Reader(s)

- Same sites as the writers — the returned tuple is immediately unpacked and
  used. A reader that obtains a stale/orphaned tuple uses dead caches for the
  rest of the call.

## Race scenario

1. Thread A calls `run()` with `device=0` for the first time, reaches line 720.
   `self.device_caches[0]` triggers `__missing__`, which calls `create_binder`.
   `create_binder` starts executing: imports modules, calls
   `get_current_target()`, builds a backend, writes `self.CompiledKernel` /
   `self.compile` / `self.ASTSource`, creates a binder, and returns
   `({}, {}, target_A, backend_A, binder_A)`. Call this tuple `T_A`.
2. Thread B calls `run()` with `device=0` concurrently. It also hits line 720.
   `defaultdict.__missing__` fires again (the key is still missing because A
   hasn't returned from the factory yet). `create_binder` runs a second time,
   producing tuple `T_B` with its own fresh empty `kernel_cache` and
   `kernel_key_cache`.
3. Both `__missing__` calls attempt `self[key] = factory_result`. One wins; the
   other's tuple is orphaned.
4. Thread A unpacks its tuple `T_A` on line 720:
   `kernel_cache, kernel_key_cache, target, backend, binder = T_A`. If `T_A`
   is the orphaned one, Thread A proceeds to compile with caches that will
   never be consulted by future calls. The compiled kernel is written into
   the orphaned `kernel_cache` at line 885 — effectively lost.
5. A later Thread C calls `run()` with `device=0`. It gets the surviving entry,
   finds no cached kernel for Thread A's specialization key, and recompiles.

**Consequence:** Silent duplicate compilation and wasted GPU resources. In
async-compile mode, the orphaned `kernel_cache` may also receive a
`finalize_compile` callback (line 877) writing to a dict that no one reads,
while the real `kernel_cache` never gets the result.

## Suggested fix

Replace the `defaultdict` with explicit locking around first-access
initialization. A per-instance lock is sufficient since the contention is per
`JITFunction` object:

```python
def __init__(self, fn, ...):
    ...
    self._device_caches = {}
    self._device_caches_lock = threading.Lock()
    ...

def _get_device_cache(self, device):
    try:
        return self._device_caches[device]
    except KeyError:
        pass
    with self._device_caches_lock:
        # Double-check after acquiring the lock.
        if device not in self._device_caches:
            self._device_caches[device] = self.create_binder()
        return self._device_caches[device]
```

All call sites (`run`, `preload`, `_do_compile`) should use
`self._get_device_cache(device)` instead of `self.device_caches[device]`.

Additionally, the `self.CompiledKernel` / `self.compile` / `self.ASTSource`
side-effect writes in `create_binder` should be moved to a one-time
initialization path (e.g., guarded by the same lock or set in `__init__`),
rather than being re-executed on every new device.
