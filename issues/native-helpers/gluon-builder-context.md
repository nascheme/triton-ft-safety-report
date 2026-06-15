# Gluon IR bindings shared MLIRContext — uniquer race (rejected)

- **Issue-Id:** FT032
- **Status:** Rejected (premise incorrect)
- **Rank:** Rejected
- **Component:** `python/src/gluon_ir.cc`

## Summary

This was originally suspected as a uniquer race: `GluonOpBuilder` holds a non-owning
`MLIRContext *` (`py::init<MLIRContext *>()`) that comes from `ir.cc`'s
`context()` factory, constructed with `MLIRContext::Threading::DISABLED`. The
claim was that the context's `StorageUniquer` is therefore unlocked, so
concurrent `*Attr::get` / `*EncodingAttr::get` / `*Type::get` calls from
multiple `GluonOpBuilder` bindings corrupt the uniquer hash maps.

**The premise is the same incorrect one as FT035. The uniquers are already
locked, so no race exists on the normal path.**

## Why the premise is wrong

Verified against the LLVM commit pinned by this build (`7f77ca0d`):

- Each `StorageUniquer` carries its own `threadingIsEnabled` flag, default
  `true`, and only skips locking when *that* flag is false.
- The construction-time `Threading` enum does not propagate to the uniquers.
  Neither `MLIRContext::MLIRContext(...)` nor `MLIRContextImpl(bool)` calls
  `disableMultithreading()` on `affineUniquer` / `attributeUniquer` /
  `typeUniquer`. Only the *runtime* `MLIRContext::disableMultithreading()`
  method does.
- `StorageUniquer`'s parametric `getOrCreate` takes a per-shard `SmartRWMutex`
  (reader then writer) whenever `threadingIsEnabled` is true.

In `ir.cc`, `disableMultithreading()` is only called on the per-compilation
context from **debug-only** code paths:

- `setupTritonDiagnosticHandler` — `if (showStacktraces)` (`ir.cc:136`)
- pass-manager IR dump — `if (haveDump)` from `MLIR_ENABLE_DUMP` (`ir.cc:1889`)
- `PassManager.run` — `if (MLIR_DISABLE_MULTITHREADING)` env (`ir.cc:1929`)
- `PassManager.run` — only when `TRITON_REPRODUCER_PATH` is set (`ir.cc:1947`)

On the normal compile path none of these fire, so the per-compilation context's
uniquers keep `threadingIsEnabled = true` and serialize all `*Attr::get` /
`*Type::get` through their per-shard mutex. The described uniquer-corruption
race does not occur.

The narrow exception is exactly the debug modes above: when one of those env
flags/diagnostics is active, `disableMultithreading()` *does* propagate and
unlock the uniquers. But those modes are an explicit opt-in to single-threaded
MLIR behavior (`MLIR_DISABLE_MULTITHREADING` literally requests it); using such
a context concurrently from multiple threads is user error, not a default-path
bug.

## Why the proposed patch was dropped

The patch switched the shared `context()` factory to `Threading::ENABLED`. It is
both unnecessary and a net negative:

- Unnecessary: the uniquers are already locked on the normal path, so it does
  not fix the stated race.
- It makes *every* per-compilation context create an owned `DefaultThreadPool`
  (≈ hardware-concurrency OS threads per compile).
- More importantly, it flips `isMultithreadingEnabled()` to true, which lets
  MLIR's pass manager run passes in parallel using that pool — behavior Triton
  deliberately disables. The patch comment asserts "Triton does not exploit
  MLIR's parallel pass execution," but enabling threading is what *allows* MLIR
  to spawn that parallelism, potentially introducing new hazards rather than
  removing one.

The patch file has been removed.
