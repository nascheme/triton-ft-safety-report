# tools

Audit of `python/triton/tools/` — offline / CLI utilities (AOT compile, link,
disassembly, external-library stub generation, memory sanitizer wrapper,
triton→gluon translator, MX float helpers, tensor descriptor, ragged TMA).

Most of this subtree is a set of CLI entry points that run once per process
(`python -m triton.tools.compile`, `link`, `gsan`, `build_extern`,
`triton_to_gluon_translator`). Those are not exercised concurrently by the
normal Triton JIT hot path and, in practice, are Tier-0 concerns for free
threading. The one real exception is `disasm.get_sass`, which is imported and
called from `compiler/compiler.py` at runtime whenever a user reads
`compiled_kernel.asm["sass"]`. Everything else is listed for completeness so a
follow-up agent can decide whether deeper investigation is warranted.

`tensor_descriptor.TensorDescriptor` is also reachable at runtime (it is
imported by `runtime/interpreter.py` and used as a public API), but it is a
plain `@dataclass` with no module-level mutable state and no lazy init, so it
does not appear in the issue list.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | Minor | `tools/disasm.py` | 1 | Module-level `@functools.lru_cache()` on `get_sass()` is shared mutable cache reachable from the compiler runtime path |
| 2 | Minor | `tools/compile.py` | 3 | `compile_kernel()` mutates process-global `sys.path` (`sys.path.insert(0, ...)`) and never restores it |
| 3 | Minor | `tools/gsan.py` | 3 | `main()` writes `triton.knobs.compilation.instrumentation_mode` and mutates `sys.argv` / `sys.path` (Tier 3 config mutation) |
| 4 | Minor | `tools/build_extern.py` | 2 | `LLVMDisassembler._ll_file` hard-codes `"/tmp/extern_lib.ll"` as a shared-path side channel |

## Triage notes

### 1. `disasm.get_sass` LRU cache (only runtime-reachable concern)

```python
# tools/disasm.py
@functools.lru_cache()
def get_sass(cubin_asm, fun=None):
    fd, path = tempfile.mkstemp()
    try:
        with open(fd, 'wb') as cubin:
            cubin.write(cubin_asm)
        sass = extract(path, fun)
    finally:
        os.remove(path)
    return sass
```

- Called from `compiler/compiler.py`: `AsmDict.__getitem__` on the `"sass"` key
  (two call sites: lines 343 and 395). That makes it a real runtime cache
  shared across threads, keyed on raw `bytes` cubin payloads.
- Per-call side effects are on per-process temp files via `tempfile.mkstemp`,
  so there is no file-name collision between concurrent calls.
- `functools.lru_cache` on free-threaded CPython uses an internal lock, so
  cache-insert races should not corrupt the cache itself; worst case is two
  threads simultaneously disassembling the same cubin and one of the two
  results being discarded. CLAUDE.md asks us to flag `@lru_cache` usage on
  shared hot paths for a follow-up review rather than guessing about the
  free-threading guarantees, so this is listed as Minor.
- The called `extract()` runs `cuobjdump` via `subprocess.check_output` and
  does pure-Python parsing on the returned bytes — no additional shared
  mutable state.

### 2. `compile.compile_kernel` mutates `sys.path`

```python
# tools/compile.py
arg_path = Path(args.path)
sys.path.insert(0, str(arg_path.parent))
spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
```

- `sys.path` is a process-global list. The insert is never undone.
- This function is reachable via `main()` (the CLI entry point) but is also
  a plain importable function; library callers could in principle call it
  concurrently.
- In practice this is a one-shot AOT compile CLI that invokes the function
  once per process, so the shared state is not raced in the typical use case.
  Listed as Minor for visibility only.

### 3. `gsan.main` mutates global configuration

```python
# tools/gsan.py
triton.knobs.compilation.instrumentation_mode = "gsan"
with torch.cuda.use_mem_pool(create_mem_pool()), _script_context(...):
    runpy.run_path(str(script_path), run_name="__main__")
```

- Tier 3 pattern (configuration mutation during compilation / launch). See
  the knobs component for the underlying race on `knobs.compilation`.
- `_script_context` additionally mutates `sys.argv` and `sys.path` and
  restores them on exit. Fine when called once from a CLI; unsafe if two
  threads called `main()` concurrently, but that is not a realistic usage.
- Listed as Minor because the entry point is CLI-only in practice.

### 4. `build_extern.LLVMDisassembler._ll_file`

```python
# tools/build_extern.py
class LLVMDisassembler:
    def __init__(self, path):
        self._path = path
        self._ll_file = "/tmp/extern_lib.ll"
```

- Hard-coded shared path: concurrent `build()` invocations on the same host
  would clobber each other's intermediate `.ll` file and can produce wrong
  stub outputs.
- This is a build-time tool for regenerating `libdevice.py` stubs. It is not
  invoked from the Triton JIT runtime. Listed for completeness.

### Items that were looked at and rejected

- **`tools/disasm.py` module-level compiled regexes** (`FLINE_RE`, `SLINE_RE`,
  `FNAME_RE`, `BRA_RE`). Compiled `re.Pattern` objects are effectively
  immutable; concurrent `match`/`findall`/`sub` are safe. Not an issue.
- **`tools/link.py` `HeaderParser`**. All state is per-instance; `main()`
  creates one parser per CLI invocation. No shared globals. Not an issue.
- **`tools/build_extern.py` `Libdevice._symbols`, `_symbol_groups`**. Populated
  once per instance via `parse_symbols` and then read. Per-instance state,
  not shared. Not an issue.
- **`tools/tensor_descriptor.TensorDescriptor`**. Plain dataclass; fields are
  written in `__post_init__` and not mutated afterwards. Typically created
  per call site and passed into a kernel launch. Not an issue.
- **`tools/mxfp.py` `MXFP4Tensor` / `MXScaleTensor`**. Pure-Python helpers
  that allocate per-call torch tensors; no module-level mutable state. Not
  an issue.
- **`tools/ragged_tma.py`**. Two `@triton.jit` helpers plus a plain Python
  factory. The `@triton.jit` objects are registered in
  `_triton_jit_function_registry` at import time, which is an import-locked
  write-once event; any per-instance races on those JITFunction objects are
  covered by the `jit` component, not here. Not an issue.
- **`tools/triton_to_gluon_translator/`**. Offline AST rewriter (CLI entry
  `_main_cli`). All state is per-invocation (`Translator`, `ordered_set`,
  `scoped_dict`, `stable_toposort`). `inline_helpers.defs` is a module-level
  `dict[str, str]` populated at import and never mutated. Not an issue.
- **`tools/compile.py` / `link.py` template file I/O**. Per-invocation
  `output_file.open("w")` writes to user-supplied output paths. Not a shared
  cache. Not an issue.
- **`build_extern.py` `subprocess.Popen(...).communicate()`** for `autopep8`
  and `isort`. Per-call subprocesses on user-supplied output paths. Not an
  issue.
