# `dtype2str` `std::unordered_map` mutated on the tensordesc specialization path

- **Issue-Id:** FT043
- **Status:** Open
- **Rank:** Critical Blocker
- **Component:** `python/src/specialize.cc`
- **Patch:** [`dtype-ptr2str-unordered-map-race.patch`](dtype-ptr2str-unordered-map-race.patch)

`dtype2str` and `dtype_ptr2str` are sibling process-global
`std::unordered_map` caches in `specialize.cc` with the same threading
hazard. They are tracked together under [FT042](dtype-ptr2str-unordered-map-race.md)
and fixed by the same patch.

`dtype2str` specifically is mutated by `specialize_tensordesc()` on the
per-argument specialization path for tensordesc / Gluon descriptor
arguments. See FT042 for the full shared-state / writer / reader / race
description and fix shape.

Note that entries are *only* inserted into the maps: `dtype2str` (like
`dtype_ptr2str`) has no calls to `erase`, `remove`, or `clear` in
`specialize.cc`. Entries are never deleted; the stored `PyObject *` values live
for the process lifetime. This insert-only property makes the two-phase miss
path in the patch safe.
