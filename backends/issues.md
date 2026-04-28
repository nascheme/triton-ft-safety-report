# backends/ Free-Threading Issues

Issues in `python/triton/backends/__init__.py`, `backends/compiler.py`, and
`backends/driver.py` affecting free-threaded Python 3.14t.

Files in scope (merged into a single component because they are thin
interface / registry code that is easier to reason about together):

- `backends/__init__.py` — backend discovery via in-tree directory walk or
  `importlib.metadata.entry_points`. Exposes the process-global
  `backends: dict[str, Backend]` registry populated at import time by
  `_discover_backends()`.
- `backends/compiler.py` — `BaseBackend` abstract class, `GPUTarget`
  dataclass, `Language` enum, and a handful of `@staticmethod`
  specialization helpers (`parse_attr`, `get_int_specialization`,
  `get_tensor_specialization`). Pure interface; no module-level mutable
  state.
- `backends/driver.py` — `DriverBase` abstract class, `GPUDriver` base,
  `Benchmarker` protocol, and tensor-descriptor helpers
  (`decompose_descriptor`, `wrap_handle_tensordesc_impl`,
  `_parse_descriptor`, `_expand_descriptor`, `expand_signature`). Pure
  helper functions plus one base class whose instance fields are set
  once in `__init__`.

This is a low-surface component. The bulk of the thread-safety story
for the things these files define lives elsewhere:

- The active/default `DriverBase` singleton and its `set_active` method
  are in `runtime/driver.py` — already audited under `runtime-driver/`.
- Each concrete backend's `driver.py` (`backends/nvidia/driver.py`
  etc.) has its own per-backend state — already audited under
  `nvidia-driver/`.
- `BaseBackend.compile`/`add_stages` is invoked through
  `compiler/compiler.py` and is covered under `compiler/`.

What remains in this component is:

1. the `backends` module-global dict (read by `compiler.py:make_backend`,
   `runtime/driver.py:_create_driver`, etc.), and
2. whatever per-instance state `BaseBackend`/`DriverBase`/`GPUDriver`
   subclasses inherit from these base files.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | Minor | backends (registry) | 3 | [`backends` dict has no in-tree writer but is reachable for out-of-tree mutation](backends-registry-mutation.md) |
| 2 | Minor | GPUDriver.__init__ | 1 | [`GPUDriver.__init__` reads `torch.cuda.*` lazily on first construction — cold-start only](gpu-driver-init-torch-attrs.md) |

No Significant or SEVERE issues identified. Both items above are
documented for completeness; the deep-dive agent may well reclassify
them as "not worth reporting."

## Triage notes

### 1. `backends` module-global dict

- **Code (`__init__.py:66`):**
  ```python
  backends: dict[str, Backend] = _discover_backends()
  ```
- **Writer:** `_discover_backends()` at import time — runs under Python's
  per-module import lock, which is preserved on free-threaded CPython.
  This is exactly the "write-once globals set during module init"
  pattern that CLAUDE.md carves out.
- **Readers:** `compiler/compiler.py:make_backend` (iterates
  `backends.values()` at line 367), `runtime/driver.py:_create_driver`
  (reads `backends[selected]` at line 14 and `backends.values()` at
  line 19). These are on the hot compile / first-launch path.
- **Grep coverage of writers:** no in-tree code writes to
  `triton.backends.backends` after import. The dict is effectively
  frozen.
- **Residual concern (Tier 3, theoretical):** the registry is a plain
  `dict`, not a `MappingProxyType` or a frozen structure. Downstream
  users (plugins, tests, notebooks) can do
  `triton.backends.backends[name] = Backend(...)` after import. If one
  thread does that while another thread is iterating `backends.values()`
  in `make_backend` or `_create_driver`, the iteration is undefined per
  `PYTHON_THREADSAFETY.md`. Also, the iteration + list-comp at
  `compiler.py:367` and `runtime/driver.py:19` is followed by
  `len(actives)`/`len(x)` checks — a concurrent insert can split the
  iteration so that a newly-registered backend is seen twice or
  missed.
- **Severity:** Minor. No in-tree code mutates `backends` post-import
  and the intended extension mechanism (entry points) runs at import
  time too. The deep-dive agent should confirm (a) that no callback
  registered by an entry point ever re-registers into `backends`, and
  (b) whether the "Tier 3 mutation" concern is worth hardening
  (e.g., wrapping the public surface in `MappingProxyType`, or
  documenting it as read-only after import).
- **Tier:** 3.
- **Suggested fix direction:** consider exposing `backends` as a
  read-only mapping (either `types.MappingProxyType(_backends)` or a
  getter function) so that mutation post-import is at least noisy.
  Alternatively, document the registry as frozen and leave the dict
  in place.

### 2. `GPUDriver.__init__` — torch attribute capture

- **Code (`driver.py:161-171`):**
  ```python
  def __init__(self):
      import torch
      self.get_device_capability = torch.cuda.get_device_capability
      try:
          from torch._C import _cuda_getCurrentRawStream
          self.get_current_stream = _cuda_getCurrentRawStream
      except ImportError:
          self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
      self.get_current_device = torch.cuda.current_device
      self.set_current_device = torch.cuda.set_device
  ```
- **Concern:** The lazy `import torch` + repeated `torch.cuda.*`
  attribute reads run once per `GPUDriver` subclass instance. If two
  threads construct the same `GPUDriver` subclass for the first time
  (torch not yet imported), they race on torch's own import side
  effects. Python's import lock serializes this.
- **Concrete consequence:** nothing. `import torch` is protected by
  the import lock, and the subsequent attribute reads are single
  attribute loads on already-initialized modules.
- **Severity:** Minor. Primarily a "benign cold-start" note.
- **Tier:** 1.
- **Suggested fix direction:** none required. The deep-dive agent
  should double-check that no torch attribute involved here (`cuda.
  get_device_capability`, `cuda.current_device`, `cuda.set_device`,
  `_cuda_getCurrentRawStream`) is lazily populated on first access in
  a way that could race; based on PyTorch source those are module-level
  functions bound at torch import, but a version bump could change
  that.

## Not worth reporting

### `BaseBackend.supports_native_tensor_specialization` class attribute

- **Code (`compiler.py:24`):** `supports_native_tensor_specialization = True`.
- **Why not:** Grep shows no assignment to this attribute outside the
  class body anywhere in `triton/python/`. Subclasses that want to
  override set it at class-definition time, which is write-once.
  Read-only process-global behavior.

### `_find_concrete_subclasses` dir/getattr walk

- **Code (`__init__.py:19-29`).**
- **Why not:** Called only from `_discover_backends()`, which runs at
  import time under the import lock. `dir()` returns a snapshot list,
  `getattr` is a single attribute load, `isinstance` / `issubclass` /
  `inspect.isabstract` are pure.

### `entry_points().select(...)` and `importlib.import_module` in `_discover_backends`

- **Why not:** Import-time only. `importlib.metadata.entry_points` has
  its own internal caching (mutated under its own internal lock) and
  is only consulted once per process when `triton.backends` is
  imported. Downstream `importlib.import_module` goes through the
  import system's own locking.

### `wrap_handle_tensordesc_impl` closure capture

- **Code (`driver.py:19-47`).**
- **Why not:** Builds `relevant_paths` locally, closes over it, and
  returns an `inner` function. The dict is read-only after the
  closure is built — no post-construction writes — so concurrent
  `inner(*args)` calls only read it and pass it into the native
  `make_tensordesc_args` helper.

### Pure helpers

- `decompose_descriptor`, `_is_descriptor`, `_parse_descriptor`,
  `_expand_descriptor`, `expand_signature`, `Benchmarker` (Protocol),
  `BaseBackend.parse_attr`, `BaseBackend.get_int_specialization`,
  `BaseBackend.get_tensor_specialization`. Pure functions over their
  arguments and module-level constants — no state.

### `GPUDriver.allocate_default_profile_scratch`

- **Code (`driver.py:177-187`).**
- **Why not:** Each call creates a fresh `torch.zeros` tensor and
  possibly a fresh `ExternalStream`. No shared mutable state on
  `self`. Any thread-safety concern about `torch.cuda.stream` /
  `record_stream` itself is PyTorch's responsibility, not Triton's.

### `@dataclass(frozen=True) class Backend` / `GPUTarget`

- **Why not:** Frozen dataclasses with no mutable fields. Sharing a
  `Backend` or `GPUTarget` instance across threads is safe by
  construction.

### `Language(Enum)`

- **Why not:** Standard-library Enum. Values are created at class
  definition; the Enum class itself has no post-definition mutation.

### `DriverBase.__init__` (empty body)

- **Why not:** Does nothing. Subclasses add their own state; audit
  their per-subclass `__init__` in the per-backend reports
  (`nvidia-driver/` etc.).

## Follow-ups for the deep-dive agent

1. Verify (by grep across the Triton ecosystem, including
   `third_party/` backend directories) whether any code path mutates
   `triton.backends.backends` after import. If yes, issue #1 should be
   re-evaluated.
2. Confirm that no subclass of `BaseBackend` overrides
   `supports_native_tensor_specialization` at runtime (e.g., in a
   test harness that flips the flag on a shared class). If yes, that
   is an instance-level or class-level mutation that may become a
   Tier 3 bug.
3. Confirm that `entry_points()` iteration does not have Triton-side
   caching that could be consulted post-import. The current code
   calls `entry_points()` exactly once; if a future refactor memoizes
   the result on `triton.backends`, that memoization would become a
   new Tier 1/3 audit target.
