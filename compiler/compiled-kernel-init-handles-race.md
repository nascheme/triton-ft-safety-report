# `CompiledKernel._init_handles` lazy handle initialization race

**Severity:** SEVERE
**Tier:** 2 (multiple threads calling the same kernel concurrently)
**File:** `python/triton/compiler/compiler.py`

## Shared state

A `CompiledKernel` instance is the reusable wrapper handed back from the
compile/cache path. It is stored in
`JITFunction.device_caches[device].kernel_cache[key]` (see
`runtime/jit.py:877,882,885`) and returned to every subsequent caller of
the same specialization. Two or more Python threads launching the same
kernel therefore race on the same instance.

The lazily-initialized fields on that instance are:

- `self._run` — the driver launcher callable (set at `compiler.py:455`)
- `self.module` — the loaded GPU module handle (set at `compiler.py:468`)
- `self.function` — the loaded GPU function handle (set at `compiler.py:468`)
- `self.n_regs`, `self.n_spills`, `self.n_max_threads` (set at
  `compiler.py:468`)

All five are `None` after `__init__` (`compiler.py:435-437`).

## Writers

`CompiledKernel._init_handles()` (`compiler.py:439-474`) is the only
writer. The initialization is **not atomic** and writes these fields in
two separate steps:

```python
def _init_handles(self):
    if self.module is not None:             # 440   guard
        return
    ...
    self._run = driver.active.launcher_cls(self.src, self.metadata)  # 455
    ...
    if knobs.runtime.kernel_load_start_hook is not None:
        knobs.runtime.kernel_load_start_hook(...)                    # 466
    self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = (
        driver.active.utils.load_binary(...))                        # 468
    ...
    if knobs.runtime.kernel_load_end_hook is not None:
        knobs.runtime.kernel_load_end_hook(...)                      # 474
```

Two things to notice:

1. **`self._run` is published at line 455, well before `self.module` /
   `self.function` are populated at line 468.**
2. **The guard at line 440 checks `self.module`, not `self._run`.** So a
   second thread arriving during the window 455..468 will *not* see the
   guard trigger, even though `self._run` is already non-`None`.

The tuple-target assignment at line 468 is itself not atomic under
free-threading: CPython lowers it to sequential `STORE_ATTR` opcodes, so
another thread can observe `self.module` set while `self.function`,
`self.n_regs`, etc. are still `None`.

## Readers

There are three entry points into the lazy-init path, and several places
that read the handles directly:

- **`CompiledKernel.run` property** (`compiler.py:476-480`):
  ```python
  @property
  def run(self):
      if self._run is None:
          self._init_handles()
      return self._run
  ```
  This is the hot-path trigger used by `JITFunction.run` at
  `runtime/jit.py:761`:
  ```python
  kernel.run(grid_0, grid_1, grid_2, stream, kernel.function,
             kernel.packed_metadata, launch_metadata,
             knobs.runtime.launch_enter_hook,
             knobs.runtime.launch_exit_hook, *bound_args.values())
  ```
  Note that the property guard is on `self._run is None`, not
  `self.module is None`. This is the critical asymmetry.

- **`CompiledKernel.__getitem__`** (`compiler.py:493-504`) calls
  `_init_handles()` on entry, then returns a `runner` closure that, when
  called, re-reads `self.function` and `self.run` (the property).

- **`CompiledKernel.launch_metadata`** (`compiler.py:482-491`) calls
  `_init_handles()` only when `knobs.runtime.launch_enter_hook is not
  None` — so the common no-hook launch path goes through the `run`
  property instead.

## Race scenarios

### Scenario A — "published-but-empty launcher" (SEVERE)

This is the worst interleaving and does not require any unusual timing.
Two threads launch the same kernel for the first time after a cache hit
in `kernel_cache`.

| Time | Thread A (`jit.py:761` → `kernel.run`)               | Thread B (`jit.py:761` → `kernel.run`)               |
|------|-------------------------------------------------------|-------------------------------------------------------|
| t0   | `run` property: `self._run is None` → True            |                                                       |
| t1   | enters `_init_handles`, guard `self.module is None`   |                                                       |
| t2   | `self._run = launcher_cls(...)` (line 455)            |                                                       |
| t3   | scheduled out                                         |                                                       |
| t4   |                                                       | `run` property: `self._run is None` → **False**       |
| t5   |                                                       | returns `self._run` (the launcher)                    |
| t6   |                                                       | `jit.py:761` passes `kernel.function` (still `None`)  |
| t7   |                                                       | launcher invoked with `function=None` → crash / UB    |

The launcher is the C++ binding generated by `make_launcher.py` /
`driver.active.launcher_cls`; it passes `function` straight to the GPU
driver API (`cuLaunchKernel` or equivalent). Passing a null function
handle is undefined behavior at best and a segfault at worst.

Thread B does *not* re-enter `_init_handles` because the `run` property
short-circuits on `self._run is not None`. Thread A's partial write is
therefore observable and there is no recovery path.

### Scenario B — guard TOCTOU, duplicate load + duplicate hooks (Significant)

Entry through `__getitem__` (or `launch_metadata` when the launch hook
is set) uses the `self.module is None` guard at line 440 rather than the
`self._run is None` check. Two threads both pass the guard and run the
full init in parallel:

- Both call `driver.active.launcher_cls(...)`.
- Both call `driver.active.utils.load_binary(...)`. Whether this is
  safe depends on the driver: CUDA's `cuModuleLoadData` is documented as
  thread-safe, but any bookkeeping layered on top in
  `python/src/driver.c` / the backend shim is not necessarily safe.
- Both invoke `knobs.runtime.kernel_load_start_hook` and
  `knobs.runtime.kernel_load_end_hook`. The hook is a user-visible
  observability callback; duplicate invocations are a real correctness
  issue for anyone counting kernel loads or recording timestamps.
- The last writer wins for `self.module`, `self.function`, etc. Any
  GPU-side state produced by the loser's `load_binary` call is leaked
  (no `cuModuleUnload`).

### Scenario C — torn tuple-unpack assignment (SEVERE)

The five-target assignment at line 468 is sequential `STORE_ATTR`:

```
self.module       = tup[0]   # STORE_ATTR
self.function     = tup[1]   # STORE_ATTR
self.n_regs       = tup[2]   # STORE_ATTR
self.n_spills     = tup[3]   # STORE_ATTR
self.n_max_threads = tup[4]  # STORE_ATTR
```

Thread A performs these in order. Thread B, racing on the entry guard
at line 440, can observe `self.module is not None` as soon as the first
`STORE_ATTR` lands, then return from `_init_handles()` and proceed to
read `self.function` (still `None`). This is the same observable
failure as Scenario A, reached through `__getitem__` /
`launch_metadata` instead of the `run` property.

### Scenario D — `raise_` partial poisoning

The `raise_` helper at lines 443-451 deep-copies the exception and
assigns `self._run = functools.partial(_raise_error, cloned_err)`. This
is intentional poisoning so future calls re-raise instead of silently
succeeding. Under free-threading, the same two-step problem applies:
Thread A hits `raise_`, poisons `self._run`, but a concurrent Thread B
that already read an unpoisoned `self._run` via the property will
launch the old launcher against the freshly-partially-initialized
state. Low priority because the outer control flow is already an
error path, but worth noting.

## Concrete consequences

1. **Crash / undefined behavior from null `function` handle** — Scenario
   A and Scenario C. The driver launcher is handed a `None` where it
   expects a GPU function pointer.
2. **Duplicate `cuModuleLoadData` and matching GPU state leak** —
   Scenario B. Depending on the backend this may or may not corrupt
   the module table.
3. **Duplicate observability hook invocations** — `kernel_load_start_hook`
   and `kernel_load_end_hook` fire twice for the same `CompiledKernel`,
   breaking tracing/metrics that assume one start/end per instance.
4. **Wrong `n_regs` / `n_spills` / `n_max_threads`** — Scenario B; the
   loser's values are overwritten but the `num_warps * warp_size`
   check at line 471 may have already run against the winner's values
   and raised or passed incorrectly.

## Suggested fix

Add a per-instance lock, keep `self._run` as the publish flag, and make
sure `self._run` is the **last** write on the success path and is also
assigned on the error path so that the existing `raise_` poisoning
contract is preserved.

### Poisoning contract in the original code

The original `raise_` helper (`compiler.py:443-451`) assigns
`self._run = functools.partial(_raise_error, cloned_err)` before raising,
so that subsequent calls via the `run` property short-circuit:

```python
@property
def run(self):
    if self._run is None:   # poisoned partial is not None → skip init
        self._init_handles()
    return self._run         # return the poisoned partial; calling it raises
```

Any fix must keep this behavior: once `_init_handles` has been entered
and failed, future calls should see `self._run` non-`None` and re-raise
from the poisoned partial without running `load_binary` or the
`kernel_load_*_hook` callbacks again.

### Fix

```python
def __init__(self, ...):
    ...
    self._init_lock = threading.Lock()
    self.module = None
    self.function = None
    self._run = None

def _init_handles(self):
    if self._run is not None:          # fast path: success or poisoned
        return
    with self._init_lock:
        if self._run is not None:
            return
        try:
            device = driver.active.get_current_device()
            launcher = driver.active.launcher_cls(self.src, self.metadata)
            max_shared = max_shared_mem(device)
            if self.metadata.shared > max_shared:
                raise OutOfResources(
                    self.metadata.shared, max_shared, "shared memory")
            if (hasattr(self.metadata, "tmem_size")
                    and self.metadata.tmem_size is not None
                    and self.metadata.tmem_size > 512):
                raise OutOfResources(
                    self.metadata.tmem_size, 512, "tensor memory")
            if knobs.runtime.kernel_load_start_hook is not None:
                knobs.runtime.kernel_load_start_hook(
                    None, None, self.name, self.metadata_group, self.hash)
            module, function, n_regs, n_spills, n_max_threads = (
                driver.active.utils.load_binary(
                    self.name, self.kernel, self.metadata.shared, device))
            warp_size = driver.active.get_current_target().warp_size
            if self.metadata.num_warps * warp_size > n_max_threads:
                raise OutOfResources(
                    self.metadata.num_warps * warp_size,
                    n_max_threads, "threads")
        except BaseException as err:
            # Poison so future calls short-circuit and re-raise cheaply.
            self._run = functools.partial(_raise_error, copy.deepcopy(err))
            raise
        # Success: assign handles first, publish self._run LAST.
        self.n_regs = n_regs
        self.n_spills = n_spills
        self.n_max_threads = n_max_threads
        self.module = module
        self.function = function
        self._run = launcher
        if knobs.runtime.kernel_load_end_hook is not None:
            knobs.runtime.kernel_load_end_hook(
                module, function, self.name, self.metadata_group, self.hash)
```

The `run` property keeps its original shape:

```python
@property
def run(self):
    if self._run is None:
        self._init_handles()
    return self._run
```

`__getitem__` and `launch_metadata` should change their direct
`_init_handles()` guard (or just call through the `run` property) so all
three entry points use `self._run` as the publish flag consistently.

### Why this works

1. **`self._run` is the single source of truth for "init attempted".**
   `self._run is None` strictly means "no thread has entered the body
   yet". Any other state (real launcher or poisoned partial) short-
   circuits the guard.
2. **Poisoning is preserved.** On the error path the `except` assigns
   `self._run` to the raising partial **before** the exception
   propagates out of the `with` block. The next caller sees
   `self._run is not None`, returns from the fast path, receives the
   partial via the `run` property, and re-raises without re-running
   `launcher_cls`, `load_binary`, or either hook.
3. **No torn publication on success.** `self._run = launcher` is the
   last write in the success branch. A concurrent reader that observes
   a non-`None` `self._run` is guaranteed to have already observed
   `self.module`, `self.function`, `self.n_regs`, `self.n_spills`, and
   `self.n_max_threads`, because all of those stores happened earlier
   on the same thread under `_init_lock`, and free-threaded CPython's
   per-object lock on `LOAD_ATTR` / `STORE_ATTR` gives the necessary
   happens-before for a reader that later loads `self._run` and then
   `self.function`.
4. **Hooks run exactly once.** Both `kernel_load_start_hook` and
   `kernel_load_end_hook` are inside the critical section, and the
   inner `if self._run is not None: return` re-check prevents a second
   entrant from re-running them even if it was already waiting on the
   lock.
5. **Duplicate `load_binary` is impossible.** The lock serializes the
   first call and the re-check short-circuits any thread that arrived
   while the first was inside the body.

### Lighter alternative (not recommended)

Change only the `run` property's guard to `self.module is None`:

```python
@property
def run(self):
    if self.module is None:
        self._init_handles()
    return self._run
```

This kills Scenario A (the published-but-empty launcher) but still
leaves Scenarios B and C (guard TOCTOU and torn tuple-unpack) and does
not preserve the `raise_` poisoning short-circuit — on the error path
`self.module` is never assigned, so every future `run` access would
re-enter `_init_handles` and re-run the failing body. The full locking
fix is the defensible answer.
