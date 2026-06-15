#!/usr/bin/env python3
"""Deterministic reproducer for FT026.

This does not require a GPU. It constructs the visible state FT026 is about:
``kernel_cache`` has a cache hit, but ``JITFunction.used_global_vals`` is still
the initial empty dict. Vulnerable Triton then iterates zero entries and skips
the global-changed safety check.

Run against an installed or built Triton:

    python issues/jit/used-global-vals-unsynchronized-read-reproducer.py

If Triton is not importable, the script tries the sibling ``../triton/python``
checkout layout used by this review workspace.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


def import_triton() -> tuple[Any, Any]:
    try:
        import triton  # type: ignore[import-not-found]
        import triton.runtime.jit as jit_mod  # type: ignore[import-not-found]
        return triton, jit_mod
    except ModuleNotFoundError:
        pass

    workspace = Path(__file__).resolve().parents[3]
    local_triton = workspace / "triton" / "python"
    if local_triton.exists():
        sys.path.insert(0, str(local_triton))

    try:
        import triton  # type: ignore[import-not-found]
        import triton.runtime.jit as jit_mod  # type: ignore[import-not-found]
        return triton, jit_mod
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise SystemExit(
            "Could not import Triton. Run this in an environment with an installed "
            "Triton, or with a built Triton checkout on PYTHONPATH.\n"
            f"Import failure: {type(exc).__name__}: {exc}"
        ) from exc


triton, jit_mod = import_triton()

CAPTURED_GLOBAL = 1


@triton.jit
def ft026_kernel(x):
    return x + CAPTURED_GLOBAL


class FakeDriver:
    def get_current_device(self):
        return "ft026-device"

    def get_current_stream(self, device):
        return "ft026-stream"


class FakeKernel:
    pass


class FakeOptions:
    def __str__(self):
        return "ft026-options"


def install_cache_hit(fn):
    options = FakeOptions()
    cache_key = "ft026-cache-key"
    kernel_cache = {cache_key: FakeKernel()}
    kernel_key_cache = {((), str(options)): cache_key}

    def binder(*args, **kwargs):
        return {}, [], options

    fn.device_caches = {
        "ft026-device": (
            kernel_cache,
            kernel_key_cache,
            object(),  # target, unused on the cache-hit path
            object(),  # backend, unused on the cache-hit path
            binder,
        )
    }


def reset_kernel_state():
    global CAPTURED_GLOBAL
    CAPTURED_GLOBAL = 1
    ft026_kernel.hash = None
    ft026_kernel.used_global_vals = {}
    install_cache_hit(ft026_kernel)


def run_cache_hit():
    return ft026_kernel.run(0, grid=(1,), warmup=True)


def demonstrate_vulnerable_state() -> bool:
    global CAPTURED_GLOBAL
    reset_kernel_state()
    CAPTURED_GLOBAL = 2

    try:
        result = run_cache_hit()
    except RuntimeError as exc:
        print("cache-hit run raised before skipping the check; FT026 appears fixed")
        print(f"  {exc}")
        return False

    if not isinstance(result, FakeKernel):
        raise AssertionError(f"expected FakeKernel cache hit, got {type(result).__name__}")
    if ft026_kernel.used_global_vals:
        raise AssertionError("expected used_global_vals to remain empty on the vulnerable path")

    print("FT026 reproduced: cache-hit run skipped the global-changed check")
    print("  used_global_vals was empty even though CAPTURED_GLOBAL changed")
    return True


def demonstrate_forced_cache_key_behavior() -> None:
    global CAPTURED_GLOBAL
    reset_kernel_state()

    # This is the proposed fix in miniature: force dependency discovery before
    # the cache lookup path can observe a cached kernel.
    ft026_kernel.cache_key
    if not ft026_kernel.used_global_vals:
        raise AssertionError("cache_key did not populate used_global_vals")

    CAPTURED_GLOBAL = 2
    try:
        run_cache_hit()
    except RuntimeError as exc:
        print("forcing cache_key first raises as expected")
        print(f"  {exc}")
        return

    raise AssertionError("expected global-changed check to raise after forcing cache_key")


def main() -> None:
    old_active = jit_mod.driver._active
    jit_mod.driver.set_active(FakeDriver())
    try:
        demonstrate_vulnerable_state()
        demonstrate_forced_cache_key_behavior()
    finally:
        jit_mod.driver._active = old_active


if __name__ == "__main__":
    main()
