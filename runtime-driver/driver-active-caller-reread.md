# Hot-path callers re-read `driver.active` across multiple statements

- **Status:** Open
- **Severity:** Significant
- **Component:** `compiler/compiler.py:453-470`,
  `runtime/jit.py:713-714,809`, `runtime/autotuner.py:125,185`

- **Shared state:** `DriverConfig._active`, swappable at runtime via
  `set_active()` / `reset_active()`.
- **Writer(s):** `set_active()` and `reset_active()` in `driver.py`.
- **Reader(s):** Hot-path functions that read `driver.active` multiple
  times in the same function body (e.g. `_init_handles` reads it four
  times).
- **Race scenario:** Thread A in `_init_handles` reads `driver.active`
  → `D1` for device and launcher_cls. Thread B calls `set_active(D2)`.
  Thread A's next read returns `D2`, so `load_binary` uses `D2` while
  `self._run` was bound to `D1`. The launcher receives a handle from the
  wrong driver.
- **Suggested fix:** Cache `driver.active` into a local at the top of
  each function that reads it more than once, and use the local
  throughout. Independent of the `DriverConfig` internal lock — each
  individual read is already atomic; the hazard is across multiple reads.
