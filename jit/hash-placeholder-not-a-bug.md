# `JITCallable.hash` placeholder visibility -- not a bug

- **Status:** Not a bug
- **Severity:** None
- **Component:** `runtime/jit.py`
- **Tier:** 2

The `cache_key` property sets a placeholder `self.hash = f"recursion:{self._fn_name}"`
(line 507) inside `_hash_lock` before computing the real hash. The concern was
that another thread could read the placeholder directly and treat it as a real
hash.

**Conclusion:** Every read of `self.hash` in the codebase is inside
`cache_key` and therefore under `_hash_lock`. No external reader exists, so
the placeholder never escapes the locked region. See README.md triage note #10
for the full investigation.
