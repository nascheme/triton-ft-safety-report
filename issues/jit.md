# runtime/jit.py Free-Threading Issues

Scope: `python/triton/runtime/jit.py`.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| FT021 | MED | function_registry | 1 | [`_triton_jit_function_registry` publishes partially-initialized `JITFunction`](jit/function-registry-race.md) |
| FT020 | HIGH | device_caches | 2 | [`JITFunction.device_caches` defaultdict auto-vivification race](jit/device-caches-race.md) |
| FT023 | MED | kernel_cache | 2 | [`kernel_cache` / `kernel_key_cache` TOCTOU causes duplicate compilation](jit/kernel-cache-toctou.md) |
| FT026 | MED | used_global_vals | 2 | [`JITCallable.used_global_vals` unsynchronized read skips global-changed safety check](jit/used-global-vals-unsynchronized-read.md) |
| FT024 | MED | pre_run_hooks | 3 | [`JITFunction.pre_run_hooks` unsynchronized iteration during concurrent mutation](jit/pre-run-hooks-unsynchronized-iteration.md) |
| FT019 | MED | async_compile | 2 | [`_do_compile` async path - `AsyncCompileMode`/`FutureKernel` races](jit/async-compile-races.md) |
| FT025 | MED | _unsafe_update_src | 2 | [`JITCallable._unsafe_update_src` unsynchronized hash invalidation](jit/unsafe-update-src-race.md) |
| FT018 | MED | add_stages_inspection_hook | 3 | [`knobs.runtime.add_stages_inspection_hook` TOCTOU in `run()`](jit/add-stages-inspection-hook-toctou.md) |

## Notes

- FT020 is the only HIGH item in this component.
- FT021 is Tier 1 because a non-Triton thread can observe the global registry
  while one Triton thread constructs a `JITFunction`.
- FT024 and FT018 are Tier 3 configuration/hook mutation during execution.
- `KernelParam.cached_property` fields, `create_binder` identical attribute
  writes, append-only registry lookups, and single-load knob reads were reviewed
  and not reported separately.
