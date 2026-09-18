"""Optional host pacing for a simulation; no system-wide timer changes."""

from __future__ import annotations

import ctypes
import math
import os
import time


class SimulationPacer:
    """Scoped one-shot waitable timer. It never advances or skips sim steps.

    Win32 API contract:
    https://learn.microsoft.com/windows/win32/api/synchapi/nf-synchapi-createwaitabletimerexw
    https://learn.microsoft.com/windows/win32/api/synchapi/nf-synchapi-setwaitabletimer
    A relative negative due time is measured in 100 ns units. No busy spinning,
    elevated scheduling priority, or global timeBeginPeriod setting is used.
    """

    def __init__(self, clock="sleep", spin_margin_s=0.0):
        if clock not in ("sleep", "windows-high-resolution"):
            raise ValueError("unknown simulation pacing clock")
        if clock == "windows-high-resolution" and os.name != "nt":
            raise ValueError("Windows high-resolution clock requires Windows")
        if not math.isfinite(spin_margin_s) or spin_margin_s < 0.0 or spin_margin_s > 0.005:
            raise ValueError("spin margin must be between 0 and 5 ms")
        self.clock = clock
        # Waiting the whole interval lets the core reach a deep idle state, and
        # the wake-up out of it showed up as isolated start lateness of 9-11 ms
        # on an otherwise idle machine.  A short spin over the final stretch
        # keeps the thread runnable and the core awake.  Zero preserves the
        # original pure-wait behaviour for every existing caller.
        self.spin_margin_s = float(spin_margin_s)
        self.handle = None
        self.closed = False

    def __enter__(self):
        if self.closed or self.handle is not None:
            raise RuntimeError("pacer context cannot be reopened")
        if self.clock == "windows-high-resolution":
            from ctypes import wintypes

            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateWaitableTimerExW.argtypes = [
                ctypes.c_void_p,
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
            ]
            self.kernel.CreateWaitableTimerExW.restype = wintypes.HANDLE
            self.kernel.SetWaitableTimer.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ctypes.c_longlong),
                wintypes.LONG,
                ctypes.c_void_p,
                ctypes.c_void_p,
                wintypes.BOOL,
            ]
            self.kernel.SetWaitableTimer.restype = wintypes.BOOL
            self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            self.kernel.WaitForSingleObject.restype = wintypes.DWORD
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.kernel.CloseHandle.restype = wintypes.BOOL
            # CREATE_WAITABLE_TIMER_HIGH_RESOLUTION; SYNCHRONIZE | TIMER_MODIFY_STATE.
            self.handle = self.kernel.CreateWaitableTimerExW(None, None, 0x2, 0x00100002)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
        return self

    def sleep_until(self, deadline):
        if self.closed:
            raise RuntimeError("pacer already closed")
        if not math.isfinite(deadline):
            raise ValueError("nonfinite simulation deadline")
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 1.0:
            raise ValueError("simulation pacer accepts only short control waits")
        if self.spin_margin_s:
            waited_until = deadline - self.spin_margin_s
            remaining = waited_until - time.perf_counter()
            if remaining > 0:
                self._wait(remaining)
            while time.perf_counter() < deadline:
                pass
            return
        self._wait(remaining)

    def _wait(self, remaining):
        if self.clock == "sleep":
            time.sleep(remaining)
            return
        if self.handle is None:
            raise RuntimeError("pacer context not entered")
        due = ctypes.c_longlong(-max(1, math.ceil(remaining * 10_000_000)))
        if not self.kernel.SetWaitableTimer(self.handle, ctypes.byref(due), 0, None, None, False):
            raise ctypes.WinError(ctypes.get_last_error())
        status = self.kernel.WaitForSingleObject(self.handle, 2000)
        if status == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        if status != 0:
            raise RuntimeError("simulation waitable timer did not signal")

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True
        if self.handle is not None:
            handle, self.handle = self.handle, None
            if not self.kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        return False
