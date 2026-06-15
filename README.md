# Triton Free-Threading Safety Audit

Tracking thread-safety issues in Triton's Python layer for Python 3.14t
(free-threading / nogil).

We are auditing the Python code under `python/` (both the pure-Python
`triton/` package and the C++ pybind11 extensions in `src/`). The C++/MLIR/LLVM
core is out of scope except where it interacts with the Python layer: GIL
release points, pybind11 bindings, and core signatures changed only to fix a
binding-layer issue.

## Start Here

- [Issue index and tier model](issues/README.md): proposed patches, audit
  progress, component links, and the canonical Tier 1/2/3 definitions.
- [Audit methodology](CLAUDE.md): thread-safety assumptions, issue format, and
  patch expectations.
- [Architecture overview](OVERVIEW.md): Triton components and how the audit
  surface is organized.
- [Python thread-safety notes](PYTHON_THREADSAFETY.md): assumptions about
  free-threaded CPython container behavior.
- [Glossary](GLOSSARY.md): terminology used throughout the report.

Tier 1-2 HIGH/MED findings that currently block the free-threading goal are
listed in the [proposed patch queue](issues/README.md#proposed-patches). LOW
findings and unsupported Tier 3 scenarios remain tracked in the component issue
files for audit completeness.
