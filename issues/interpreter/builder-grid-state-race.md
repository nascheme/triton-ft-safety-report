# `interpreter_builder.grid_idx` / `grid_dim` trampled by concurrent kernel runs

- **Status:** Open
- **Severity:** HIGH
- **Component:** `runtime/interpreter.py` (`InterpreterBuilder`, `GridExecutor`)
- **Tier:** 2

- **Shared state:** the module-level singleton `interpreter_builder =
  InterpreterBuilder()` (interpreter.py near the bottom of the file). Its
  `grid_dim` and `grid_idx` attributes are mutated per grid-iteration and
  are the source of truth for `tl.program_id()` / `tl.num_programs()`
  inside an interpreted kernel.
- **Writer(s):** `GridExecutor.__call__` calls
  `interpreter_builder.set_grid_dim(*grid)` once, then
  `interpreter_builder.set_grid_idx(x, y, z)` in the inner grid loop just
  before invoking `self.fn(**args)`.
- **Reader(s):** `InterpreterBuilder.create_get_program_id(axis)` /
  `create_get_num_programs(axis)` read `self.grid_idx[axis]` /
  `self.grid_dim[axis]` from inside the interpreted kernel body (via the
  patched `tl.program_id` / `tl.num_programs`).
- **Race scenario:** Thread A is running a kernel with grid `(8, 1, 1)`
  and has just set `grid_idx = (5, 0, 0)`. Before A's kernel body reads
  `program_id(0)`, Thread B (running a different kernel) calls
  `set_grid_dim(16, 1, 1)` and `set_grid_idx(12, 0, 0)`. A's kernel now
  observes `program_id(0) == 12` and `num_programs(0) == 16`. Silent
  wrong-kernel execution; if A's grid is smaller than B's, A may also
  index off the end of its own buffers.
- **Suggested fix:** move per-call state off the module-level singleton.
  Either store `grid_dim` / `grid_idx` in a `ContextVar` /
  `threading.local`, or serialize `GridExecutor.__call__` with a
  process-wide lock. (Concurrent interpreter use is listed as
  pragmatically out of scope in `README.md`; a documented single-thread
  assertion is the cheapest fix.)
