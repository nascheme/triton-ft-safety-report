# runtime/cache.py Free-Threading Issues

Issues in `python/triton/runtime/cache.py` affecting free-threaded Python 3.14t.

File in scope:

- `runtime/cache.py` — cache manager abstractions (`CacheManager`,
  `FileCacheManager`, `RemoteCacheManager`, `RedisRemoteCacheBackend`),
  factory helpers `get_cache_manager` / `get_override_manager` /
  `get_dump_manager`, key helpers `make_so_cache_key` / `get_cache_key`,
  and the `@functools.lru_cache`-wrapped `triton_key()` self-hash.

This is a thin filesystem-coordinated module. Most hazards are delegated
to POSIX atomic replace and to `@functools.lru_cache`'s internal
synchronization, which are both safe under free-threading. The
interesting free-threading issues are caller-side — in `compiler.py`
and `runtime/jit.py` — and are covered by the corresponding writeups in
`compiler/` and `jit/`. This file itself has only minor residual
concerns.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| 1 | Minor | FileCacheManager.put | 2 | [`os.removedirs(temp_dir)` ascent can nuke shared parent dirs under concurrent cleanup](put-removedirs-ascent.md) |
| 2 | Minor | RemoteCacheManager / RedisRemoteCacheBackend | 3 | [Non-atomic multi-read of `knobs.cache.redis.*` / `knobs.cache.remote_manager_class` during construction](remote-cache-construction-reread.md) |
| 3 | Minor | triton_key | 1 | [`triton_key()` `@functools.lru_cache` cold-start / filesystem-dependent hashing](triton-key-lru-cache.md) |
| 4 | Minor | FileCacheManager.get_file | 2 | [`get_file` returns a path that can disappear before the caller opens it](get-file-caller-toctou.md) |

All four are Minor. No Significant or SEVERE issues identified in this
file.

## Triage notes

### 1. `FileCacheManager.put` — `os.removedirs(temp_dir)` ascent

- **Code (cache.py:115-125):**
  ```python
  temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
  os.makedirs(temp_dir, exist_ok=True)
  temp_path = os.path.join(temp_dir, filename)
  with open(temp_path, mode) as f:
      f.write(data)
  os.replace(temp_path, filepath)
  os.removedirs(temp_dir)
  ```
- **Why the write itself is safe:** UUID + PID make `temp_dir` unique
  per call, so concurrent puts never touch each other's staging dir.
  `os.replace` is POSIX-atomic, so the final `filepath` is either the
  old or the new file content, never a partial write. This matches
  CLAUDE.md's "filesystem atomic replace by itself is not a bug"
  carve-out.
- **Residual concern:** `os.removedirs` does not stop at `temp_dir`.
  It removes the leaf, then walks *up*, removing every empty parent
  until a non-empty one or an error. The parents are shared across
  every put that shares a `cache_dir` and, further up, across every
  cache key in the process (`~/.triton/cache`).
- **Race scenarios:**
  1. Normal case: at the moment `os.removedirs` walks up past
     `temp_dir`, `cache_dir` contains the just-replaced file plus
     possibly other temp dirs from concurrent puts. Non-empty, removedirs
     stops. Safe.
  2. Pathological case: an external cleaner (or a test harness) has
     just removed every file under `cache_dir` while thread A was
     between its `os.replace` and `os.removedirs`. A's `os.replace`
     wrote `cache_dir/file`, but the cleaner just removed it. A's
     removedirs then walks up through `cache_dir` (now empty),
     `~/.triton/cache` (possibly empty), `~/.triton` (possibly empty).
     Concurrent thread B is mid-`os.makedirs(cache_dir_B/tmp_B,
     exist_ok=True)` for a different key; its intermediate `mkdir` for
     `~/.triton/cache` can race with A's parallel `rmdir` of the same
     directory. Python's `os.makedirs(..., exist_ok=True)` handles
     `EEXIST` but will surface other errors, so in principle B can see
     a transient `FileNotFoundError` / `ENOENT` on an ancestor that A
     just removed, even though A's ancestor removal is "benign".
- **Severity:** Minor. Real users do not run external `rm -rf` against
  `~/.triton/cache` while Triton is active; the scenario requires
  either a buggy cleanup thread or a tight test harness. Still worth a
  note because the pattern is slightly anti-social: a single put's
  cleanup can remove directories that a concurrent put's
  `os.makedirs` is about to rely on.
- **Tier:** 2 (multiple threads caching the same or adjacent kernels).
- **Suggested fix direction:** replace `os.removedirs(temp_dir)` with
  `os.rmdir(temp_dir)` — the leaf is the only directory this function
  is responsible for. Leave cache directory cleanup to an explicit
  tool. (This also avoids the minor I/O cost of repeatedly trying to
  remove non-empty parents under load.)

### 2. `RedisRemoteCacheBackend` / `RemoteCacheManager` — non-atomic construction

- **Code (cache.py:148-155):**
  ```python
  def __init__(self, key):
      import redis
      self._key = key
      self._key_fmt = knobs.cache.redis.key_format
      self._redis = redis.Redis(
          host=knobs.cache.redis.host,
          port=knobs.cache.redis.port,
      )
  ```
  And (cache.py:170-186) `RemoteCacheManager.__init__` reads
  `knobs.cache.remote_manager_class`, then constructs the backend, then
  constructs a `FileCacheManager(key, override=override, dump=dump)`
  which reads `knobs.cache.dump_dir` / `knobs.cache.override_dir` /
  `knobs.cache.dir` depending on the mode.
- **Concern:** Each individual `knobs.*` attribute read is atomic
  (single dict lookup), but the sequence is not. If another thread
  mutates any one of the knobs mid-construction (Tier 3), the
  resulting backend captures a mix of old and new configuration:
  new `key_format` with old `host`, new `remote_manager_class` paired
  with old `dir`, etc.
- **Severity:** Minor. Knob mutation during a compile is a documented
  unsupported workflow, but the construction is already "pull a
  consistent snapshot" by intent. Consequence is wrong-backend
  routing for the lifetime of the constructed manager.
- **Tier:** 3.
- **Suggested fix direction:** in each constructor, snapshot the knobs
  into locals up front, then use the locals. Example:
  ```python
  def __init__(self, key):
      fmt = knobs.cache.redis.key_format
      host = knobs.cache.redis.host
      port = knobs.cache.redis.port
      ...
  ```
  This bounds the inconsistency window to one `__init__` call.
- **Out of scope:** the `redis.Redis` client object itself. redis-py
  documents its client as thread-safe via an internal connection pool,
  so once constructed concurrent use is fine.

### 3. `triton_key()` `@functools.lru_cache`

- **Code (cache.py:279-312):**
  ```python
  @functools.lru_cache()
  def triton_key():
      ...
      with open(__file__, "rb") as f:
          contents += [hashlib.sha256(f.read()).hexdigest()]
      ...
  ```
- **Concern:** Called from `get_cache_key` (cache.py:316) and from
  `autotuner.py:184`. Two threads that reach `triton_key()` on a cold
  cache will both walk the filesystem, hash `libtriton.so` and every
  `.py` in `compiler/`, `backends/`, `language/`. `functools.lru_cache`
  in free-threaded CPython is internally synchronized (PEP 703
  compatibility update), so the *cache state* is safe and only one
  result is ultimately stored. But under free-threading, the lock
  surface may allow duplicate computation before the first result is
  stored.
- **Severity:** Minor. Worst case is duplicate cold-start hashing
  (seconds, not correctness). Safe-by-construction for the cache entry.
- **Tier:** 1 (bring-up).
- **Suggested fix direction:** none required. If duplicate cold-start
  work becomes a latency issue, wrap the first-use path in an explicit
  double-checked lock. Do not attempt to downgrade to a module-global
  variable — the current `lru_cache` gives correct cache semantics.
- **Orthogonal note:** `triton_key()` reads files from the Triton
  install tree (cache.py:285-311). If those files are modified *during
  process lifetime*, the cached key is stale — this is the same
  behavior as under the GIL and is not a free-threading issue.

### 4. `FileCacheManager.get_file` — caller TOCTOU

- **Code (cache.py:62-71):**
  ```python
  def has_file(self, filename) -> bool:
      ...
      return os.path.exists(self._make_path(filename))

  def get_file(self, filename) -> Optional[str]:
      if self.has_file(filename):
          return self._make_path(filename)
      return None
  ```
- **Concern:** The returned path can disappear between
  `os.path.exists` and the caller's `open(path)`. Callers:
  - `build.py:138-146` handles the race explicitly — wraps the load
    in `try: ... except (RuntimeError, ImportError): warn(...)` and
    continues to a fresh build. Safe.
  - `compiler.py:265` uses `get_group` instead of `get_file` for the
    metadata sidecar, so the TOCTOU window is closed by atomic replace
    of the group file.
  - `compiler.py:334` (`fn_override_manager.get_file(ir_filename)`)
    reads a user-placed override file; if that file vanishes between
    `get_file` and the subsequent `parse(full_name, ...)` inside
    `compiler.py`, `parse()` raises. Low probability (override files
    are typically stable), but possible.
  - `autotuner.py:193-198` opens the file inside a plain `with open`
    — a concurrent `cache.put` that overwrites via `os.replace` is
    atomic, so the reader sees either the old or the new complete
    JSON. Safe.
- **Severity:** Minor. Current callers either handle the race
  explicitly (build.py) or rely on atomic replace to bound the damage.
- **Tier:** 2.
- **Suggested fix direction:** document the API contract ("return value
  may refer to a file that no longer exists") explicitly, or provide a
  `get_file_contents(filename)` helper that does the open under a
  try/except and returns `Optional[bytes]`. Present callers do not
  demand this change.

## Not worth reporting

- **`FileCacheManager.__init__` calling `os.makedirs(cache_dir,
  exist_ok=True)`** (cache.py:45, 55). Idempotent; `exist_ok=True`
  handles every ordering of concurrent `makedirs`/`rmdir`.

- **`FileCacheManager.put` concurrent writes of the same filename.**
  `os.replace` is POSIX-atomic. Two threads writing identical data
  to the same filename both succeed; the final file is one version
  (last write wins). Callers in this codebase always derive the data
  from a deterministic key, so the two versions are bit-identical in
  practice.

- **`FileCacheManager.put` `assert self.lock_path is not None`
  (cache.py:108).** `lock_path` is only non-None when the manager was
  created in `dump` or default mode (where `cache_dir` was also
  created). The assertion documents that invariant; it does not
  encode a runtime lock. Under free-threading it behaves the same as
  under the GIL.

- **`FileCacheManager.get_group` child-path existence checks**
  (cache.py:89-91). The group file is the last thing written in a
  successful put sequence (caller convention). A reader that sees a
  group file sees all its children because the children were put
  before the group; `os.path.exists` per child is a correctness
  check against partial cache deletion by external tooling, not a
  TOCTOU guard against our own writers.

- **`make_so_cache_key` and `get_cache_key` purity** (cache.py:269-277,
  315-317). Both are pure functions over their arguments; no module
  state.

- **`RedisRemoteCacheBackend.get` / `put`**. redis-py documents
  `redis.Redis` as thread-safe via its internal connection pool.
  Concurrent calls on a single backend instance are the library's
  responsibility.

- **`_base32` purity** (cache.py:249-251). Pure function; no state.

- **Module-level `from triton import __version__, knobs`**. Write-once
  at import time under the import lock.

- **`get_cache_manager` / `get_override_manager` / `get_dump_manager`
  each constructing a fresh `FileCacheManager` per call** (cache.py:254-266).
  These helpers do not retain state; each call creates an independent
  manager. There is no process-global manager object whose mutable
  fields could race. Per-call construction does read
  `knobs.cache.manager_class` (single atomic attribute read) — a
  mid-call mutation of that knob would affect the next call but not
  split a single construction.
