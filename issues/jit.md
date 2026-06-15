# runtime/jit.py Free-Threading Issues

Scope: `python/triton/runtime/jit.py`.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT020 | Critical Blocker | device_caches | [`JITFunction.device_caches` defaultdict auto-vivification race](jit/device-caches-race.md) |
| FT021 | Blocker | function_registry | [`_triton_jit_function_registry` publishes partially-initialized `JITFunction`](jit/function-registry-race.md) |
| FT023 | Blocker | kernel_cache | [`kernel_cache` / `kernel_key_cache` TOCTOU causes duplicate compilation](jit/kernel-cache-toctou.md) |
| FT026 | Blocker | used_global_vals | [`JITCallable.used_global_vals` unsynchronized read skips global-changed safety check](jit/used-global-vals-unsynchronized-read.md) |
| FT019 | Blocker | async_compile | [`_do_compile` async path - `AsyncCompileMode`/`FutureKernel` races](jit/async-compile-races.md) |
| FT025 | Blocker | _unsafe_update_src | [`JITCallable._unsafe_update_src` unsynchronized hash invalidation](jit/unsafe-update-src-race.md) |
| FT024 | Deferred | pre_run_hooks | `JITFunction.run()` iterates `pre_run_hooks` without snapshotting, so concurrent hook mutation can skip, duplicate, or raise during hook execution. Deferred because mutating hooks while launching is unsupported. |
| FT018 | Deferred | add_stages_inspection_hook | `run()` reads `knobs.runtime.add_stages_inspection_hook` across multiple steps, so concurrent hook mutation can change cache-key behavior mid-call. Deferred because mid-flight hook mutation is outside the current support goal. |

## Notes

- FT020 is the only Critical Blocker item in this component.
- FT021 is a current-goal Blocker because a non-Triton thread can observe the
  global registry while one Triton thread constructs a `JITFunction`.
- FT024 and FT018 are Deferred configuration/hook mutation during execution.
- `KernelParam.cached_property` fields, `create_binder` identical attribute
  writes, append-only registry lookups, and single-load knob reads were reviewed
  and not reported separately.
