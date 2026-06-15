# tools/

Scope: `python/triton/tools/`.

Most files here are offline or CLI utilities. The runtime-reachable concern is
`tools/disasm.py:get_sass`, used when a caller asks for `"sass"` in a compiled
kernel's `asm` dict.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | LOW | `tools/disasm.py` | 1 | `@functools.lru_cache()` on runtime-reachable `get_sass()` |
| 2 | LOW | `tools/compile.py` | 3 | `compile_kernel()` mutates process-global `sys.path` and never restores it |
| 3 | LOW | `tools/gsan.py` | 3 | `main()` writes `triton.knobs.compilation.instrumentation_mode` and mutates `sys.argv` / `sys.path` |
| 4 | LOW | `tools/build_extern.py` | 2 | `LLVMDisassembler._ll_file` hard-codes `/tmp/extern_lib.ll` |

## Notes

- No MED or HIGH issues were found.
- `get_sass()` uses per-call temp files; the `lru_cache` is the only shared
  runtime state.
- The CLI utilities are listed for completeness but are not normal JIT hot-path
  code.
