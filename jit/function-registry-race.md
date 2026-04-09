# `_triton_jit_function_registry` publishes partially-initialized `JITFunction`

- **Status:** Unreported
- **Severity:** Significant
- **Component:** runtime/jit.py
- **Tier:** 1
- **Source reports:** [README.md](README.md) (issue #1)

## Description

`JITFunction.__init__` inserts `self` into the module-level
`_triton_jit_function_registry` dict (line 781) before finishing
initialization. Attributes set after the insertion include `self.params`
(783), `self.device_caches` (790), `self.kernel` (794), `self.debug` /
`self.noinline` (795-796), `self.arg_names` / `self.constexprs` (800-801),
and `self.pre_run_hooks` (804).

A second thread that calls `preload()` can resolve the key via
`_decode_constant` (lines 831-832), obtain the half-built instance, and
pass it into the compile path. Any access to the missing attributes raises
`AttributeError`.

This is a publish-before-init bug, not a dict-corruption bug. The GIL does
not prevent it: the interpreter can switch threads between any two
bytecodes, so a thread switch after the `__setitem__` at line 781 but
before line 783 exposes the same half-built object to a reader. Free-
threading simply makes the window wider.

## Race scenario

1. Thread A enters `JITFunction.__init__`, executes line 781 (registry now
   holds the half-built instance), and is preempted before line 783.
2. Thread B calls `preload(specialization_data)` where the serialized data
   contains `{'jit_function': 'mymod:my_kernel'}`. `_decode_constant`
   finds the key and returns the half-built instance.
3. Downstream code accesses `.params`, `.device_caches`, or
   `.pre_run_hooks` on the returned object — `AttributeError`.

## Suggested fix

Move the registry insertion to the very end of `__init__`:

```python
def __init__(self, fn, version=None, ...):
    ...
    self.pre_run_hooks = []

    # Must be last: makes the instance visible to other threads.
    _triton_jit_function_registry[
        f"{self.module}:{self.fn.__qualname__}"
    ] = self
```
