# `DriverConfig.active` lazy init / `set_active` ordering race

- **Status:** Open
- **Patch:** `driver-default-lazy-init-race.patch` (resolved by the same `DriverConfig` lock that fixes the `default` lazy init race)
- **Severity:** Significant
- **Component:** `runtime/driver.py:36-46`
- **Tier:** 3

- **Shared state:** `DriverConfig._active` on the module-level singleton.
  Read by `driver.active` on every hot path.
- **Writer(s):** `active` getter lazy path (line 39), `set_active()`
  (line 43), `reset_active()` (line 46).
- **Reader(s):** `active` getter, from every hot path (`compiler.py`,
  `jit.py`, `autotuner.py`, testing, tools).
- **Race scenario:** Thread A enters `active`, sees `self._active is None`,
  and is about to write `self._active = self.default`. Thread B calls
  `driver.set_active(custom_driver)`. Thread A then overwrites B's
  explicit override with the default, silently losing the user's
  `set_active` call. Wrong driver selection on the hot path.
- **Suggested fix:** Use the same lock as the `default` race to cover
  `active`, `set_active`, and `reset_active`. The `active` getter should
  single-load `self._active` into a local before the `None` check.
  See [driver-active-caller-reread.md](driver-active-caller-reread.md)
  for the separate caller-side multiple-read hazard.
