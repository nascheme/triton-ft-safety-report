# runtime/driver.py Free-Threading Issues

Scope: `python/triton/runtime/driver.py`, especially the process-global
`driver = DriverConfig()` object read from compiler, JIT, autotuner, tools, and
tests.

## Issues

| # | Rank | Component | Issue |
| --- | --- | --- | --- |
| FT041 | Critical Blocker | upstream import | [Upstream segfault on Python 3.13t during Triton import](runtime-driver/seg-fault-gh-6721.md) |
| FT040 | Blocker | DriverConfig.default | [`DriverConfig.default` lazy init race creates duplicate driver instances](runtime-driver/driver-default-lazy-init-race.md) |
| FT039 | Deferred | DriverConfig.active | `DriverConfig.active` lazy initialization can race with `set_active`, exposing stale or overwritten active-driver state. Deferred because changing the active driver while other threads compile or launch is unsupported. |
| FT038 | Deferred | hot-path callers | Several hot-path callers read `driver.active` more than once, so concurrent active-driver mutation can mix driver objects within one operation. Deferred because it requires unsupported driver/backend switching mid-flight. |

FT041 is an upstream symptom. FT040 is a plausible contributing race, but the
upstream issue does not prove causality.

## Notes

- FT040 is the actionable current-goal race: duplicate backend driver
  construction.
- FT039 and FT038 are Deferred because they require driver/backend selection
  mutation while another thread is compiling or launching.
- Caller fixes should snapshot `driver.active` once when multiple driver
  attributes are needed.
