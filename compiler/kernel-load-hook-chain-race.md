# `kernel_load_start_hook` / `kernel_load_end_hook` — `HookChain` iteration race

- **Files:** `python/triton/knobs.py`, `python/triton/compiler/compiler.py`
- **Severity:** Significant
- **Tier:** 3 (configuration mutation concurrent with kernel load)

## Shared state

`HookChain.calls: list[F]` on the four process-global `HookChain`
slots declared in `runtime_knobs` (`knobs.py:468-471`):

```python
launch_enter_hook:      HookChain[LaunchHook]     = HookChain()
launch_exit_hook:       HookChain[LaunchHook]     = HookChain(reversed=True)
kernel_load_start_hook: HookChain[InitHandleHook] = HookChain()
kernel_load_end_hook:   HookChain[InitHandleHook] = HookChain(reversed=True)
```

`HookChain` (`knobs.py:406-424`) has no synchronization:

```python
class HookChain(Generic[F]):
    def __init__(self, reversed: bool = False):
        self.calls: list[F] = []
        self.reversed = reversed

    def add(self, func: F) -> None:
        if func not in self.calls:
            self.calls.append(func)

    def remove(self, func: F) -> None:
        if func in self.calls:
            self.calls.remove(func)

    def __call__(self, *args, **kwargs):
        for call in self.calls if not self.reversed else reversed(self.calls):
            call(*args, **kwargs)
```

This writeup focuses on `kernel_load_start_hook` / `kernel_load_end_hook`
(reached from `compiler.py`), but the bug is a property of `HookChain`
and also affects `launch_enter_hook` / `launch_exit_hook` on the launch
path (`compiler.py:502`, `runtime/jit.py:762`).

## Readers (iteration sites)

`HookChain.__call__` iterates `self.calls`. Reached from:

- `compiler.py:466` — `kernel_load_start_hook(...)` in
  `CompiledKernel._init_handles`, on first use of a compiled kernel.
- `compiler.py:474` — `kernel_load_end_hook(...)` in the same function,
  after `driver.active.utils.load_binary`.

## Writers (mutation sites)

`HookChain.add` / `HookChain.remove` on the same global instances. In
the tree:

- `third_party/proton/proton/hooks/hook.py:101,126` — Proton adds on
  `start`, removes on `stop`.
- `test/unit/runtime/test_launch.py:94-146` — tests add and remove
  around launches.

These are the intended API for attaching/detaching profiling and
instrumentation at runtime.

## Race A — iterator observes undefined element sequence

1. Thread A enters `_init_handles` on a cold `CompiledKernel` and
   calls `kernel_load_start_hook(...)`. `HookChain.__call__` begins
   `for call in self.calls:`.
2. Thread B calls `kernel_load_start_hook.add(hook_X)` (e.g.
   `profile.start()`).
3. The element sequence observed by A's iterator is undefined (per
   https://docs.python.org/dev/library/threadsafety.html): a
   previously-registered hook may be skipped or visited twice, and
   `hook_X` may or may not fire for the in-flight call. The
   `reversed=True` variant (`kernel_load_end_hook`) has the same
   issue via `list_reverseiterator`.

`remove` during iteration has the same consequence — `list.remove`
shifts tail elements, and A's index can land past a live element.

## Race B — check-then-act in `add` / `remove`

`add`'s `if func not in self.calls: self.calls.append(func)` is a
classic check-then-act: two concurrent `add(same_func)` calls can
both observe absence and both append, duplicating the hook. `remove`
is symmetric: two concurrent `remove(same_func)` calls can both pass
the membership check, and the second `list.remove` raises
`ValueError`. Minor relative to race A, but part of the same fix.

## Consequence

1. **Silently skipping or double-invoking a kernel-load hook.** Proton
   and similar profilers use `kernel_load_end_hook` to capture
   per-kernel binary metadata (module/function pointers, kernel
   name). A skipped invocation means the profiler never sees a
   loaded kernel; a double invocation skews counters or double-
   registers. No crash, no wrong kernel launched — the damage is
   confined to the hook invocation sequence.

2. **Process-global scope.** Unlike `pre_run_hooks` (per-`JITFunction`),
   these HookChains are global. Any concurrent `_init_handles` on any
   `CompiledKernel` races any concurrent `add` / `remove`. Window
   widens with kernel count.

3. **Compounds with issue #1.** The `CompiledKernel._init_handles`
   race already causes duplicate entry, so each duplicate hook
   invocation is additionally vulnerable to race A.

4. **Launch path also affected.** `launch_enter_hook` /
   `launch_exit_hook` are also `HookChain` instances and are invoked
   on every kernel launch (`compiler.py:502`, `runtime/jit.py:762`).
   Same bug, higher frequency. Fixing `HookChain` repairs all four
   slots at once.

## Suggested fix

Copy-on-write in `HookChain`, same pattern as the
[`pre_run_hooks` fix](../jit/pre-run-hooks-unsynchronized-iteration.md):

```python
import threading

class HookChain(Generic[F]):
    def __init__(self, reversed: bool = False):
        self._lock = threading.Lock()
        self.calls: list[F] = []   # immutable after publish
        self.reversed = reversed

    def add(self, func: F) -> None:
        with self._lock:
            if func in self.calls:
                return
            new_calls = list(self.calls)
            new_calls.append(func)
            self.calls = new_calls

    def remove(self, func: F) -> None:
        with self._lock:
            if func not in self.calls:
                return
            new_calls = list(self.calls)
            new_calls.remove(func)
            self.calls = new_calls

    def __call__(self, *args, **kwargs):
        calls = self.calls   # single load — stable snapshot
        for call in calls if not self.reversed else reversed(calls):
            call(*args, **kwargs)
```

The published list is never mutated again; writers copy-append-rebind
under the lock. `__call__` snapshots `self.calls` into a local and
iterates freely. The lock is never held across hook invocation
(which can be slow and can re-enter Triton). No changes required at
the call sites in `compiler.py` or `runtime/jit.py`.

## Related

- [`jit/pre-run-hooks-unsynchronized-iteration.md`](../jit/pre-run-hooks-unsynchronized-iteration.md)
  — same iteration race on per-`JITFunction` hooks; same fix
  pattern.
- [`compiler/compiled-kernel-init-handles-race.md`](compiled-kernel-init-handles-race.md)
  — issue #1, whose duplicate `_init_handles` entry compounds this.
