# `do_bench` `@cached_property` duplicate execution

- **Status:** Open
- **Severity:** Minor
- **Component:** `python/triton/runtime/autotuner.py`

- **Shared state:** `self.do_bench` backed by `@cached_property` (line 122),
  stored in `self.__dict__`.
- **Writer(s):** First access triggers the getter, which calls
  `driver.active.get_benchmarker()` or returns `self._do_bench`.
- **Reader(s):** `_bench()` at line 163 (`self.do_bench(kernel_call, ...)`).
- **Race scenario:** Two threads access `self.do_bench` concurrently before
  it is cached. Both execute the getter and both write to `self.__dict__`.
  `@cached_property` lost its internal lock in Python 3.12. The getter is
  pure — `get_benchmarker()` returns the same function object — so duplicate
  execution produces an identical result.
- **Consequences:** Redundant call to `get_benchmarker()`. No correctness
  impact.
- **Suggested fix:** No fix required. Accept the benign duplicate or replace
  with an eagerly initialized attribute in `__init__` if the driver is
  available at construction time.
