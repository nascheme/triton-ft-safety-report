# `linear_layout.cc` shared MLIRContext — uniquer race (rejected)

- **Issue-Id:** FT035
- **Status:** Rejected (premise incorrect)
- **Rank:** Rejected
- **Component:** `python/src/linear_layout.cc`

## Summary

This was originally suspected as a race on the StorageUniquer of the process-wide
`MLIRContext` shared by `getLinearLayoutContext()`. The claim was that the
context is built with `MLIRContext::Threading::DISABLED`, so its uniquers take
no internal locks, making concurrent `mlir::StringAttr::get(ctx, …)` from
`identity_1d` / `strided_1d` / `zeros_1d` / `from_bases` / `apply` corrupt the
uniquer hash maps.

**The premise is false. The uniquers are already locked, so no race exists.**

## Why the premise is wrong

`StorageUniquer` locking is governed by *each uniquer's own*
`threadingIsEnabled` flag, which defaults to `true`. The construction-time
`Threading` enum does **not** reach the uniquers. Verified against the LLVM
commit pinned by this Triton build (`7f77ca0d`):

- `MLIRContext::MLIRContext(const DialectRegistry&, Threading)` only forwards
  the flag to `MLIRContextImpl(bool threadingIsEnabled)`. Its body contains no
  call to `disableMultithreading()` and none to the uniquers'
  `disableMultithreading()`.
- `MLIRContextImpl(bool)` only sets its own `threadingIsEnabled` and
  conditionally creates the thread pool. It does **not** propagate the flag to
  `affineUniquer` / `attributeUniquer` / `typeUniquer`.
- Only the runtime method `MLIRContext::disableMultithreading()` propagates into
  the uniquers:
  ```cpp
  impl->affineUniquer.disableMultithreading(disable);
  impl->attributeUniquer.disableMultithreading(disable);
  impl->typeUniquer.disableMultithreading(disable);
  ```
- `StorageUniquerImpl` declares `bool threadingIsEnabled = true;`, and the
  parametric `getOrCreate` only skips locks when that flag is false:
  ```cpp
  if (!threadingIsEnabled)
    return getOrCreateUnsafe(shard, lookupKey, ctorFn);  // unlocked
  // otherwise: SmartScopedReader / SmartScopedWriter on shard.mutex
  ```

Triton's `ir.context()` factory (`ir.cc:322-325`) only constructs
`MLIRContext(Threading::DISABLED)` and never calls `disableMultithreading()` on
it; `getLinearLayoutContext()` (`linear_layout.cc:19-28`) just reuses that
factory. So the shared context's `attributeUniquer` / `typeUniquer` keep
`threadingIsEnabled = true`. Concurrent `StringAttr::get` / `IntegerAttr::get` /
`RankedTensorType::get` are serialized by the uniquer's per-shard
`SmartRWMutex`. The "unlocked `unordered_map` corruption" scenario cannot occur.

The construction-time `Threading::DISABLED` flag controls only (a) the
context-level `isMultithreadingEnabled()` flag (which gates `parallelFor`,
`getThreadPool()`, etc. — not uniquing) and (b) whether an owned
`DefaultThreadPool` is created. It is unrelated to uniquer locking.

(The other `disableMultithreading()` calls in `ir.cc` — lines 136, 1889, 1929,
1947 — act on the per-compilation parsing contexts, a different object than this
shared linear-layout context, so they do not change this analysis.)

## Why the proposed patch was dropped

The patch called `ctx->enableMultithreading()` on the shared context. The
technique is the correct way to *re-enable* uniquer locking, but here it changes
nothing for correctness (the uniquers were already enabled) while spawning an
owned `DefaultThreadPool` (≈ hardware-concurrency OS threads) on a
process-lifetime context that never runs parallel MLIR work — a small net
negative. The patch file has been removed.
