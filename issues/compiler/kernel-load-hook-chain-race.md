# `kernel_load_start_hook` / `kernel_load_end_hook` — `HookChain` iteration race

- **Status:** Open
- **Severity:** MED
- **Component:** `python/triton/knobs.py` (`HookChain`),
  `python/triton/compiler/compiler.py`
- **Tier:** 3
- **Patch:** [`kernel-load-hook-chain-race.patch`](kernel-load-hook-chain-race.patch)

- **Shared state:** `HookChain.calls: list[F]` on the process-global
  `kernel_load_start_hook` / `kernel_load_end_hook` instances declared
  on `RuntimeKnobs` in `knobs.py`.
- **Writer(s):** `HookChain.add` / `.remove` — called by Proton
  (`hook.py`) and tests to attach/detach profiling at runtime.
- **Reader(s):** `HookChain.__call__` iterates `self.calls` — reached
  from `CompiledKernel._init_handles` in `compiler.py`.
- **Race scenario:** Thread A iterates `self.calls` in `__call__`.
  Thread B calls `.add(hook)` or `.remove(hook)`, mutating the same
  list. Iteration under concurrent mutation has undefined element
  sequence: a registered hook may be skipped or visited twice.
  Secondary: `add`'s check-then-act
  (`if func not in self.calls: self.calls.append(func)`) allows
  duplicate hooks under concurrent adds.
- **Suggested fix:** Copy-on-write: writers copy the list under a
  lock and rebind `self.calls`; `__call__` snapshots `self.calls`
  into a local before iterating. Fixes all four `HookChain` slots
  at once.

Same pattern as [`jit/pre-run-hooks-unsynchronized-iteration.md`](../jit/pre-run-hooks-unsynchronized-iteration.md).
Also affected: `launch_enter_hook` / `launch_exit_hook` on the launch
path in `compiler.py` and `runtime/jit.py`.
