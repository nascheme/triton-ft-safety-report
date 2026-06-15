# runtime/driver.py Free-Threading Issues

Scope: `python/triton/runtime/driver.py`, especially the process-global
`driver = DriverConfig()` object read from compiler, JIT, autotuner, tools, and
tests.

## Issues

| # | Severity | Component | Tier | Issue |
|---|----------|-----------|------|-------|
| FT040 | MED | DriverConfig.default | 1 | [`DriverConfig.default` lazy init race creates duplicate driver instances](runtime-driver/driver-default-lazy-init-race.md) |
| FT039 | MED | DriverConfig.active | 3 | [`DriverConfig.active` lazy init / `set_active` ordering race](runtime-driver/driver-active-set-active-race.md) |
| FT038 | MED | hot-path callers | 3 | [Hot-path callers re-read `driver.active` across multiple statements](runtime-driver/driver-active-caller-reread.md) |
| FT041 | HIGH | upstream import | 1 | [Upstream segfault on Python 3.13t during Triton import](runtime-driver/seg-fault-gh-6721.md) |

FT041 is an upstream symptom. FT040 is a plausible contributing race, but the
upstream issue does not prove causality.

## Notes

- FT040 is the actionable Tier 1 race: duplicate backend driver construction.
- FT039 and FT038 are Tier 3 because they require driver/backend selection
  mutation while another thread is compiling or launching.
- Caller fixes should snapshot `driver.active` once when multiple driver
  attributes are needed.
