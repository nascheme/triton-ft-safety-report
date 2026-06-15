# runtime/jit.py Free-Threading Issues

Scope: `python/triton/runtime/jit.py`.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT021 | Blocker | function_registry | [`_triton_jit_function_registry` publishes partially-initialized `JITFunction`](jit/function-registry-race.md) |
| FT020 | Critical Blocker | device_caches | [`JITFunction.device_caches` defaultdict auto-vivification race](jit/device-caches-race.md) |
| FT023 | Blocker | kernel_cache | [`kernel_cache` / `kernel_key_cache` TOCTOU causes duplicate compilation](jit/kernel-cache-toctou.md) |
| FT026 | Blocker | used_global_vals | [`JITCallable.used_global_vals` unsynchronized read skips global-changed safety check](jit/used-global-vals-unsynchronized-read.md) |
| FT024 | Deferred | pre_run_hooks | [`JITFunction.pre_run_hooks` unsynchronized iteration during concurrent mutation](jit/pre-run-hooks-unsynchronized-iteration.md) |
| FT019 | Blocker | async_compile | [`_do_compile` async path - `AsyncCompileMode`/`FutureKernel` races](jit/async-compile-races.md) |
| FT025 | Blocker | _unsafe_update_src | [`JITCallable._unsafe_update_src` unsynchronized hash invalidation](jit/unsafe-update-src-race.md) |
| FT018 | Deferred | add_stages_inspection_hook | [`knobs.runtime.add_stages_inspection_hook` TOCTOU in `run()`](jit/add-stages-inspection-hook-toctou.md) |

## Notes

- FT020 is the only Critical Blocker item in this component.
- FT021 is a current-goal Blocker because a non-Triton thread can observe the
  global registry while one Triton thread constructs a `JITFunction`.
- FT024 and FT018 are Deferred configuration/hook mutation during execution.
- `KernelParam.cached_property` fields, `create_binder` identical attribute
  writes, append-only registry lookups, and single-load knob reads were reviewed
  and not reported separately.
