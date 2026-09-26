"""Temporary Windows compat shim for the fresh-checkout test ONLY.

Upstream commit c58cf54 introduced os.sched_getaffinity calls that have no
win32 guard (AttributeError on Windows). This monkeypatch (applied before
importing motrix_rl) provides a POSIX-shaped fallback via ctypes
GetProcessAffinityMask so the training pipeline can run for validation.

Reported upstream; remove this once a Windows guard lands.
"""

import os
import sys

if sys.platform == "win32" and not hasattr(os, "sched_getaffinity"):
    import ctypes

    def _sched_getaffinity(mask_pid: int = 0) -> set[int]:
        kernel32 = ctypes.windll.kernel32
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        handle = kernel32.GetCurrentProcess()
        if not kernel32.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            return set(range(os.cpu_count() or 1))
        return {i for i in range(64) if (process_mask.value >> i) & 1}

    os.sched_getaffinity = _sched_getaffinity  # type: ignore[attr-defined]

if sys.platform == "win32" and not hasattr(os, "sched_setaffinity"):
    def _sched_setaffinity(mask_pid: int, cpus: set[int]) -> None:  # noqa: ARG001
        pass  # best-effort: pinning skipped on Windows

    os.sched_setaffinity = _sched_setaffinity  # type: ignore[attr-defined]
