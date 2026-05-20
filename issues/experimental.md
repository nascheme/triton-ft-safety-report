# experimental

Audit of `python/triton/experimental/`. Two subpackages:

- `experimental/gluon/` — Gluon DSL: language types, layout classes, and a
  `GluonJITFunction` subclass of `JITFunction`. Largely stateless
  class/module definitions; all runtime machinery (hot-path state, compile
  cache, etc.) is inherited from the core `jit` component.
- `experimental/gsan/` — Global Memory Sanitizer. Pure-Python glue plus a
  native `gsan_allocator` module that is JIT-built on first use, a pluggable
  CUDA allocator, a torch symmetric-memory rendezvous protocol, and stream
  synchronization helpers that compile Triton kernels on the fly.

Unlike `tools`, the gsan code is *runtime-reachable* once gsan is enabled (it
hooks torch's CUDA allocator and executes kernels on every process-group
barrier). So its module-level caches and knobs mutations are legitimate
free-threading concerns.

The gluon subtree has essentially no new shared mutable state worth
reporting; it subclasses `JITFunction` so its per-instance caches inherit
whatever the `jit` component finds. Listed here only for completeness.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | Significant | `experimental/gsan/_stream_sync.py` | 3 | `_compile_without_gsan()` mutates global `knobs.compilation.instrumentation_mode` via `knobs.compilation.scope()`; concurrent compilations on other threads will observe the cleared value |
| 2 | Significant | `experimental/gsan/symmetric_memory.py` | 2 | `rendezvous()` does check-then-act on `_RENDEZVOUS_CACHE` (WeakValueDictionary) and drives a Unix-socket FD-exchange protocol; two threads rendezvousing the same (ptr, storage, group) key can double-import peer FDs |
| 3 | Significant | `experimental/gsan/symmetric_memory.py` | 2 | `_RUNTIME_BOOTSTRAP_CACHE[key]` returns a plain `set` that is later read-modified-written (`peer in …`, `.update(…)`) across concurrent `rendezvous()` calls |
| 4 | Minor | `experimental/gsan/_allocator.py` | 1 | Three `@functools.lru_cache()` functions (`_load_gsan_module`, `_compile_gsan_allocator`, `get_allocator`) do lazy native-module compile + CUDA allocator init |
| 5 | Minor | `experimental/gsan/_stream_sync.py` | 1 | `_runtime_state_layout`, `_compiled_sync_kernel` `@functools.lru_cache()` on hot sync path |
| 6 | Minor | `experimental/gsan/symmetric_memory.py` | 1 | `_get_mem_pool` `@functools.lru_cache()` on first-`empty()` path |
| 7 | Minor | `experimental/gsan/_utils.py` | 2 | Module-level `_DLPACK_STATE: dict[int, object]` keeps DLPack metadata alive; written from Python, popped from a C callback (deleter) |

## Triage notes

### 1. `_compile_without_gsan` mutates a global knob (Tier 3)

```python
# experimental/gsan/_stream_sync.py
@contextmanager
def _compile_without_gsan():
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = ""
        yield
```

- Used to wrap `_compiled_sync_kernel.warmup(...)` inside
  `@functools.lru_cache() _compiled_sync_kernel(...)` and to wrap the kernel
  launch in `synchronize_process_group_barrier()`.
- `knobs.compilation` is a process-global singleton. While the `scope()`
  context manager saves and restores the flag, it is *not* thread-local.
- Scenario: Thread A runs `synchronize_process_group_barrier()` and enters
  `_compile_without_gsan()`, setting `instrumentation_mode = ""`. Thread B
  concurrently compiles a user kernel and reads `instrumentation_mode`. It
  sees the empty string and produces a non-instrumented compile on what was
  supposed to be a gsan run — silent correctness bug for the user kernel.
- This is Tier 3 (configuration mutation concurrent with compile) and is a
  direct consequence of the global `knobs.compilation` state; see the
  `knobs` component for the underlying structural issue. Called out here
  because gsan is the caller that actually exploits the scope during normal
  operation, not just configuration.

### 2. `_RENDEZVOUS_CACHE` check-then-act

```python
# experimental/gsan/symmetric_memory.py
_RENDEZVOUS_CACHE: weakref.WeakValueDictionary[_RendezvousCacheKey, GSanSymmetricMemoryHandle] = weakref.WeakValueDictionary()
...
cached = _RENDEZVOUS_CACHE.get(cache_key)
if cached is not None and not cached._closed:
    return cached
_RENDEZVOUS_CACHE.pop(cache_key, None)
...
_RENDEZVOUS_CACHE[cache_key] = handle
```

- `cache_key = (base_ptr, storage_key, id(process_group))` — easy to collide
  when two threads call `rendezvous()` with the same tensor/group.
- Two threads can both miss, both execute the full rendezvous protocol
  (export FDs, do an `all_gather_object`, open a Unix socket, exchange FDs,
  import peers), and both write to the cache. The second write overwrites
  the first; the first handle is then leaked (its `close()` would
  double-free peer allocations).
- Worse, the protocol itself uses an FD-per-peer; having two concurrent
  sockets on the same token collides (the token is broadcast once per
  call, so both calls have *different* tokens and therefore different
  socket paths — the protocol per se does not corrupt, but the cache does
  end up with two concurrently-live handles to the same storage).
- `GSanSymmetricMemoryHandle.close()` also mutates `_RENDEZVOUS_CACHE` via
  `.pop()`, possibly from `__del__` on any thread.
- Should be guarded with a lock around the check-then-act window, or use a
  per-key idempotency primitive.

### 3. `_RUNTIME_BOOTSTRAP_CACHE` read-modify-write on a shared set

```python
# experimental/gsan/symmetric_memory.py
_RUNTIME_BOOTSTRAP_CACHE: dict[_RuntimeBootstrapCacheKey, set[int]] = {}
...
runtime_bootstrapped_peers = _RUNTIME_BOOTSTRAP_CACHE.setdefault(runtime_bootstrap_cache_key, set())
peers_needing_runtime_bootstrap = {
    peer for peer in range(world_size)
    if peer != rank and peer not in runtime_bootstrapped_peers
}
...
if peers_needing_runtime_bootstrap:
    runtime_bootstrapped_peers.update(peers_needing_runtime_bootstrap)
```

- `setdefault` on the outer dict is a single op and is fine. But the
  returned `set` is shared across calls for the same `(process_group,
  device)` key.
- Two threads rendezvousing on the same key each read the set
  (`peer not in runtime_bootstrapped_peers`), each may decide the same
  peers need bootstrap, and each then calls `runtime_bootstrapped_peers
  .update(...)`. The set ops themselves are single-op safe under free
  threading, but the *decision* ("this peer still needs a runtime-state
  FD sent") is racy: both threads will send runtime-state FDs to the same
  peer, doubling `import_runtime_state_handle` calls.
- Needs a lock around the read-decide-update window or a per-key guard.

### 4–6. `@functools.lru_cache()` lazy-init helpers

Per CLAUDE.md, `@lru_cache` is shared mutable state that is worth a
follow-up. `functools.lru_cache` is internally locked in free-threaded
CPython, so the cache dict itself is fine; the concern is *duplicate work*
under a first-call race:

- `_allocator._load_gsan_module` (line 14) compiles a C++ module on first
  use — two threads calling before the cache is populated could both
  invoke `compile_module_from_file`, doing duplicate subprocess compilation
  and duplicate disk writes of the `.so`.
- `_allocator._compile_gsan_allocator` / `_allocator.get_allocator` — wrap
  the above.
- `_stream_sync._runtime_state_layout` (line 23) — constructs a DLPack
  torch view of the runtime-state region on first use.
- `_stream_sync._compiled_sync_kernel` (line 44) — first call runs
  `_synchronize_vector_clocks_kernel.warmup(...)` under the
  `_compile_without_gsan()` scope (see #1). Two concurrent first calls
  therefore also nest the knobs mutation.
- `symmetric_memory._get_mem_pool` (line 42) — first call per device
  index runs `create_mem_pool()` (which itself goes through
  `get_allocator()` in #4).

All listed Minor; consequence is duplicate compile/work, not corruption.

### 7. `_DLPACK_STATE` module dict written from Python, popped from C deleter

```python
# experimental/gsan/_utils.py
_DLPACK_STATE: dict[int, object] = {}

@_DLManagedTensorDeleter
def _dl_managed_tensor_deleter(dl_managed_tensor):
    _DLPACK_STATE.pop(ctypes.addressof(dl_managed_tensor.contents), None)

class _DLPackCudaPtrView:
    def __dlpack__(self, stream=None):
        ...
        _DLPACK_STATE[dl_managed_tensor_ptr] = self
        return _PyCapsule_New(...)
```

- Purpose: keep the ctypes `_DLManagedTensor` (and its owning view) alive
  until PyTorch drops the imported tensor and invokes the deleter.
- Writer: Python code on the thread consuming the view. Reader: the
  `_dl_managed_tensor_deleter` C callback, which is called by PyTorch and
  may run on any thread (including a CUDA cleanup thread).
- Both operations are single dict ops (`d[k] = v`, `d.pop(k, None)`) which
  are safe on free-threaded CPython. Listed as Minor for visibility: if
  this dict gains iteration or check-then-act mutation later (e.g. a size
  cap, eviction, or a `"key in d"` guard), the safety disappears.

### Items looked at and rejected

- **`experimental/gluon/_runtime.py` `GluonJITFunction(JITFunction)`**.
  Adds `is_gluon()` and tweaks `create_binder()` to swap in a
  `GluonASTSource`; no new shared mutable state. The actual per-instance
  caches (`device_caches`, hash lock, etc.) and the global
  `_triton_jit_function_registry` belong to the `jit` component, not here.
- **`experimental/gluon/_runtime.GluonASTSource.make_ir`**. Builds an MLIR
  module from `ir.builder(context)`; each call creates its own context /
  module. No shared state.
- **`experimental/gluon/language/*`**. `_core.py`, `_layouts.py`,
  `_semantic.py`, `_standard.py`, `_math.py`, and per-arch submodules
  (`nvidia/hopper.py`, `amd/cdna*/`, `amd/gfx1250/`, etc.) are class and
  JIT-builtin definitions. `__triton_builtin__` markers are set on
  functions at import time. No caches, no lazy init, no module-level
  mutables. Not an issue.
- **`experimental/gluon/nvidia/hopper.py` `TensorDescriptor`,
  `TensorDescriptorIm2Col`**. Plain classes; no module state.
- **`experimental/gsan/_allocator.py` `_THIS_DIR` / `_GSAN_SOURCE_PATH`**.
  Write-once module constants. Not an issue.
- **`experimental/gsan/_testing.py`**. Thin reexports of
  `libtriton.gsan_testing` decoders. No Python-side mutable state.
- **`experimental/gsan/_testing_utils.py`**. Skimmed; test helpers only —
  not on the runtime path. (Not expanded in this pass; a deep-dive agent
  may want to confirm.)
- **`experimental/gsan/symmetric_memory.py` Unix-socket rendezvous**.
  Each `rendezvous()` call allocates its own `socket.socket`, its own
  `received_fds` / `pending` dicts, and a token-derived socket path. All
  per-call state. The concern there is the two shared module-level caches
  (#2, #3), not the socket dance itself.
- **`experimental/gsan/src/` (GSanAllocator.cc / GSanLibrary.cu /
  GSan.h)**. Native / CUDA sources. Out of scope per CLAUDE.md: MLIR /
  LLVM / CUDA kernels are only flagged when they touch Python-visible
  state. The C++ side uses its own rw-locks (`rwLockAcquireRead` etc.)
  inside device code.
- **`atexit.register _clear_mem_pool_cache`**. Runs once at interpreter
  shutdown; not a thread-safety concern.
