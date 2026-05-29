# native-helpers

Lower-frequency native pybind11 bindings:

- `python/src/linear_layout.cc` — `LinearLayout` Python wrapper
- `python/src/gluon_ir.cc` — Gluon IR builder bindings, layout conversion
  helpers, and standalone module-level helpers (`compute_tmem_reg_layout`,
  `make_cga_layout`, `get_amd_mfma_scale_layout`, `get_amd_wmma_scale_layout`,
  `get_layout_view`)

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| FT035 | - | `linear_layout.cc` | 2 | [Shared MLIRContext, linear_layout (rejected)](native-helpers/shared-mlir-context.md) |
| FT034 | MED | `linear_layout.cc` | 2 | [`__imul__` mutates a shared `LinearLayout` Python object in place](native-helpers/imul-shared-mutation.md) |
| FT031 | MED | `linear_layout.cc` | 2 | [`get_shared_view` / `get_distributed_view` use `const_cast` to call non-const methods on a shared `LinearLayout`](native-helpers/const-cast-views.md) |
| FT032 | - | `gluon_ir.cc` | 2 | [Shared MLIRContext, gluon_ir (rejected)](native-helpers/gluon-builder-context.md) |
| FT033 | LOW | `gluon_ir.cc` | 1 | [`layoutToGluon` static `GluonLayouts` cache: first-call import latency, leaked refs](native-helpers/gluon-layouts-static.md) |

## Triage notes

### `linear_layout.cc` — `__imul__` and view methods (issues 2, 3)

`__imul__` mutates the LHS `LinearLayout` in place. If the same
Python `LinearLayout` instance is shared between threads, this is a classic
write-vs-write race on a non-atomic C++ value. Likely a low-probability bug
because `LinearLayout` instances are usually computed and then used
read-only, but it is on the public Python surface.

`get_shared_view` and `get_distributed_view` use
`const_cast<LinearLayout &>(self)` to call non-const helpers in
`mlir::triton::gpu`. Whether these helpers actually mutate the layout is an
implementation detail of `getSharedLayoutStr` / `getDistributedLayoutStr`. If
they do mutate, two threads invoking the same view on the same shared layout
race. If they do not, the `const_cast` is just a const-correctness
work-around. Worth a deeper read of those helpers to confirm.

### `gluon_ir.cc` — `GluonLayouts` static handles (issue 5)

`layoutToGluon` holds:

```cpp
static GluonLayouts layouts;
```

The constructor imports several Python modules and stores `py::handle`
references (`.release()`-ed) to layout classes. The handles are held for the
process lifetime, deliberately leaked.

Magic-static initialization is thread-safe per C++11. Concerns are minor:

1. The lambda runs Python C API on the first caller. That caller pays the
   import cost and (briefly, while holding init) blocks any other thread that
   reaches `layoutToGluon`. With the GIL gone, the magic-static guard still
   serializes the init itself.
2. The handles are raw `PyObject*` pointers stored without owning a
   reference. Because they were `.release()`-ed they have leaked refcounts,
   so they outlive the process — fine in practice.

This pattern is on the boundary of what CLAUDE.md flags as a write-once
global. Listed here as LOW because the lambda runs Python C API code under
the magic-static lock, which is a slightly unusual interleaving point, but no
concrete corruption scenario has been identified.

### Why the MLIRContext story does NOT matter the way originally assumed

**Correction (2026-05-28).** The original triage assumed that constructing the
context with `MLIRContext::Threading::DISABLED` disables locking in MLIR's
storage uniquer (the interner behind `StringAttr::get`, `IntegerAttr::get`,
`RankedTensorType::get`, …). That assumption is **wrong**, verified against the
LLVM commit pinned by this build (`7f77ca0d`):

- Uniquer locking is governed by each `StorageUniquer`'s own `threadingIsEnabled`
  flag, which defaults to `true`.
- The construction-time `Threading` enum does not reach the uniquers. Neither
  `MLIRContext::MLIRContext(...)` nor `MLIRContextImpl(bool)` propagates it to
  `affineUniquer` / `attributeUniquer` / `typeUniquer`. Only the *runtime*
  method `MLIRContext::disableMultithreading()` does.
- Triton's `ir.context()` factory only constructs with `Threading::DISABLED`
  and never calls `disableMultithreading()` on the resulting context.

So the uniquers keep `threadingIsEnabled = true` and serialize `*Attr::get`
through their per-shard `SmartRWMutex`. `Threading::DISABLED` here only affects
the context-level `isMultithreadingEnabled()` flag (gating `parallelFor`, etc.)
and whether an owned thread pool is created — not uniquing.

Consequence: the supposed uniquer-corruption root cause does not exist.

- **FT035** (`linear_layout.cc`) is rejected on this basis; see
  [shared-mlir-context.md](native-helpers/shared-mlir-context.md).
- **FT032** (`gluon_ir.cc`, builder context) rests on the *same* premise and
  needs re-examination on the same grounds — see the gluon section below.

Issues that do not depend on the uniquer-locking claim (e.g. FT034 in-place
`__imul__` mutation, FT031 `const_cast` views) are unaffected by this
correction.

### `linear_layout.cc` — process-wide context (issue 1, REJECTED)

This was originally reported as the most prominent free-threading hazard in the
file. It is rejected — see the corrected MLIRContext note above and
[shared-mlir-context.md](native-helpers/shared-mlir-context.md). The shared
context's uniquers remain locked, so the concurrent `StringAttr::get` race does
not occur. The description below is retained for context only.

The function-local static inside `getLinearLayoutContext`:

```cpp
mlir::MLIRContext *getLinearLayoutContext() {
  static PyObject *ctxObject = []() {
    py::module irMod = py::module::import("triton._C.libtriton.ir");
    py::object ctx = irMod.attr("context")();
    return ctx.release().ptr();
  }();
  return py::cast<mlir::MLIRContext *>(py::handle(ctxObject));
}
```

The C++11 magic-static initialization itself is thread-safe. The hazard is that
the resulting `MLIRContext` is **shared by every `LinearLayout` static
constructor and method** (`identity_1d`, `strided_1d`, `zeros_1d`, `from_bases`,
`apply`). Every one of those calls `mlir::StringAttr::get(ctx, …)` on this
shared, threading-disabled context.

This is distinct from the per-compilation MLIRContext that user code obtains
from `triton._C.libtriton.ir.context()` and then threads through
`triton.compiler.compiler.compile`. The linear-layout binding intentionally
keeps a separate process-wide context so that `LinearLayout` Python objects
can outlive any user compilation context. The trade-off is that **every**
call into the binding routes through one shared context.

### `gluon_ir.cc` — builder context (issue 4, REJECTED)

**Rejected on the same grounds as FT035.** Per the corrected MLIRContext note
above, the per-compilation context's uniquers stay locked
(`threadingIsEnabled = true`) on the normal compile path, so the `*Attr::get`
uniquer race does not occur. Verified: the only callers of
`disableMultithreading()` on this context are debug-only paths —
`if (showStacktraces)` (`ir.cc:136`), `MLIR_ENABLE_DUMP` (`ir.cc:1889`),
`MLIR_DISABLE_MULTITHREADING` (`ir.cc:1929`), and `TRITON_REPRODUCER_PATH`
(`ir.cc:1947`) — each an explicit single-threaded opt-in, not concurrent use.
See [gluon-builder-context.md](native-helpers/gluon-builder-context.md) for the
full write-up and why the `Threading::ENABLED` patch was dropped (it also
enables MLIR parallel pass execution that Triton deliberately disables). The
original description follows for context only.

`GluonOpBuilder` does not own its `MLIRContext`; it receives a pointer from
the Python wrapper (`py::init<MLIRContext *>()`). All
attribute/type construction in this file (e.g.
`buildCgaLayoutAttr`, `get_distributed_linear_layout`, `get_blocked_layout`,
`get_padded_shared_layout`, etc.) calls `*Attr::get(ctx, …)` on that context.

Two threads using two distinct builders that share the same underlying
context still race on the uniquer. The defensive (and apparently intentional)
counter-example is the cluster of module-level helpers
`compute_tmem_reg_layout`, `make_cga_layout`, `get_amd_mfma_scale_layout`,
`get_amd_wmma_scale_layout`, and `get_layout_view`: each constructs a fresh
**stack-local** `MLIRContext(MLIRContext::Threading::DISABLED)` and uses it
only inside that call. Those are safe.

This issue is fundamentally an MLIR-context-sharing problem rooted in
`ir.cc`. It is listed here because the gluon bindings are one of its larger
fan-out points: a single `GluonOpBuilder` method call can issue many
`StringAttr::get` and `*EncodingAttr::get` operations.



### Items considered and not reported

- **Local stack-allocated `MLIRContext` in module-level helpers** —
  `compute_tmem_reg_layout`, `make_cga_layout`,
  `get_amd_mfma_scale_layout`, `get_amd_wmma_scale_layout`,
  `get_layout_view`. Each builds its own context, so two callers do not share
  uniquer state. Safe.
- **`DialectRegistry registry;` in those module helpers.** Stack-local. Safe.
- **`GluonOpBuilder` per-instance state (builder, lastLoc).** Concurrent use
  of the same builder from two threads would be a Python-level mistake; not
  a free-threading-specific bug.
- **`isConvertLayoutTrivial`** calls `minimalCvtLayout` which
  uses the source/destination types' shared context. The race story is the
  same as issue 4; not reported separately.
- **`hasVerifier` SFINAE helpers and `printDiagStr`.** Pure compile-time /
  pure-function. No shared state.
