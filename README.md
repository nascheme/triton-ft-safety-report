# Triton Free-Threading Safety Audit

Tracking thread-safety issues in Triton's Python layer for Python 3.14t
(free-threading / nogil).

We are auditing the Python code under `python/` (both the pure-Python
`triton/` package and the C++ pybind11 extensions in `src/`). The C++/MLIR/LLVM
core is out of scope except where it interacts with the Python layer: GIL
release points, pybind11 bindings, and core signatures changed only to fix a
binding-layer issue.

## Start Here

- [Issue index and rank model](issues/README.md): proposed patches, rank
  counts, component links, and the canonical triage definitions.
- [Audit methodology](CLAUDE.md): thread-safety assumptions, issue format, and
  patch expectations.
- [Architecture overview](OVERVIEW.md): Triton components and how the audit
  surface is organized.
- [Python thread-safety notes](PYTHON_THREADSAFETY.md): assumptions about
  free-threaded CPython container behavior.
- [Glossary](GLOSSARY.md): terminology used throughout the report.

Current-goal **Critical Blocker** and **Blocker** findings are listed in the
[patch queue](issues/README.md#patch-queue). Deferred and Low findings remain
as compact summaries in the component tables.
