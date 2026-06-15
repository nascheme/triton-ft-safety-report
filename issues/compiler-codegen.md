# compiler-codegen/ Free-Threading Issues

Scope: `python/triton/compiler/code_generator.py` plus the empty
`compiler/make_launcher.py`.

This pass looked for additional front-end lowering issues beyond the filed
[`compiler/code-generator-gscope-iteration-race.md`](compiler/code-generator-gscope-iteration-race.md).
No new confirmed issue was promoted from this sweep.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| 1 | Blocker | code_generator | [`gscope` iteration race in `CodeGenerator.__init__` and `call_JitFunction`](compiler/code-generator-gscope-iteration-race.md) |
| 2 | Candidate | code_generator | `module_map` value-module attribute reads during `gscope` rewrite |
| 3 | Candidate | code_generator | `inspect.signature(fn)` on user callables during call lowering |
| 4 | Candidate | code_generator | `warnings.warn` / `warn_explicit` during compile |
| 5 | Candidate | code_generator | `JITFunction._src` rebind during `parse()` |
| 6 | Candidate | code_generator | Builder attribute storage on `ir.builder` |
| 7 | Candidate | code_generator | Lazy intra-function imports of semantic classes |
| 8 | Candidate | code_generator | Backend-supplied `module_map` / `codegen_fns` identity across calls |

## Notes

- Item #1 is already filed and linked from [`compiler.md`](compiler.md).
- Items #2-#8 are not confirmed findings. They are retained only as
  follow-up prompts if this area gets another deep dive.
- Module/class literal tables in `code_generator.py` are populated at import or
  class definition time and treated as read-only.
- `make_launcher.py` is empty.
