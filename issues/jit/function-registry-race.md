# `_triton_jit_function_registry` publishes partially-initialized `JITFunction`

- **Status:** Open
- **Severity:** MED
- **Component:** `runtime/jit.py`
- **Tier:** 1
- **Patch:** [`function-registry-race.patch`](function-registry-race.patch)

- **Shared state:** `_triton_jit_function_registry` (module-level dict) and
  the `JITFunction` instance being constructed
- **Writer(s):** `JITFunction.__init__` inserts `self` into the registry
  before setting `self.params`, `self.device_caches`, `self.kernel`,
  `self.arg_names` / `self.constexprs`, and `self.pre_run_hooks`.
- **Reader(s):** `preload()` resolves instances via `_decode_constant` and
  passes them into the compile path.
- **Race scenario:** Thread A enters `__init__`, executes the registry
  insert, and is preempted before assigning the remaining attributes.
  Thread B calls `preload()`, finds the half-built instance in the
  registry, and accesses `.params` or `.device_caches` -> `AttributeError`.
  This is a publish-before-init bug; it exists under the GIL too (thread
  switch between bytecodes), but free-threading widens the window.
- **Suggested fix:** Move the registry insertion to the end of `__init__`,
  after all attributes are set.
