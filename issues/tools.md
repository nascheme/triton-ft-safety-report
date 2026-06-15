# tools/

Scope: `python/triton/tools/`.

Most files here are offline or CLI utilities. The runtime-reachable concern is
`tools/disasm.py:get_sass`, used when a caller asks for `"sass"` in a compiled
kernel's `asm` dict.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| 1 | Low | `tools/disasm.py` | `@functools.lru_cache()` on runtime-reachable `get_sass()` |
| 2 | Low | `tools/compile.py` | `compile_kernel()` mutates process-global `sys.path` and never restores it |
| 3 | Low | `tools/gsan.py` | `main()` writes `triton.knobs.compilation.instrumentation_mode` and mutates `sys.argv` / `sys.path` |
| 4 | Low | `tools/build_extern.py` | `LLVMDisassembler._ll_file` hard-codes `/tmp/extern_lib.ll` |

## Notes

- No Critical Blocker or Blocker issues were found.
- `get_sass()` uses per-call temp files; the `lru_cache` is the only shared
  runtime state.
- The CLI utilities are listed for completeness but are not normal JIT hot-path
  code.
