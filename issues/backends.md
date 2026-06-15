# backends/ Free-Threading Issues

Scope: `python/triton/backends/__init__.py`, `backends/compiler.py`, and
`backends/driver.py`.

These files are mostly interface and registry code. The active/default driver
singleton is covered in [`runtime-driver.md`](runtime-driver.md), and concrete
NVIDIA state is covered in [`nvidia-driver.md`](nvidia-driver.md).

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| 1 | Low | backends registry | `backends` is populated at import time but remains mutable for out-of-tree callers |
| 2 | Low | GPUDriver.__init__ | Cold-start reads of `torch.cuda.*` during base driver construction |

## Notes

- No Critical Blocker or Blocker issues were found in this component.
- The in-tree `backends` registry has no post-import writer, so the residual
  concern is Deferred external mutation while another thread iterates it.
- `GPUDriver.__init__` snapshots torch CUDA helper functions. It is a
  cold-start observation only; concrete backend initialization is handled in
  the backend-specific audit.
- Pure helpers, frozen dataclasses/enums, abstract base classes, and import-time
  discovery are not reported.
