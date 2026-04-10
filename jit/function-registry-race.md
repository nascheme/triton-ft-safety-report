# `_triton_jit_function_registry` publishes partially-initialized `JITFunction`

- **Status:** Open
- **Severity:** Significant
- **Component:** `runtime/jit.py`

- **Shared state:** `_triton_jit_function_registry` (module-level dict) and
  the `JITFunction` instance being constructed
- **Writer(s):** `JITFunction.__init__` inserts `self` at line 781, before
  setting `self.params` (783), `self.device_caches` (790), `self.kernel` (794),
  `self.arg_names` / `self.constexprs` (800-801), `self.pre_run_hooks` (804)
- **Reader(s):** `preload()` resolves instances via `_decode_constant` (lines
  831-832) and passes them into the compile path
- **Race scenario:** Thread A enters `__init__`, executes the registry insert
  at line 781, and is preempted before line 783. Thread B calls `preload()`,
  finds the half-built instance in the registry, and accesses `.params` or
  `.device_caches` -> `AttributeError`. This is a publish-before-init bug;
  it exists under the GIL too (thread switch between bytecodes), but
  free-threading widens the window.
- **Suggested fix:** Move the registry insertion to the end of `__init__`,
  after all attributes are set.
