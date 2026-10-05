"""Angle-domain mapping: pulse-defined revolutions to resampled vibration."""

from __future__ import annotations

import numpy as np

from .errors import AnalysisError

POINTS_PER_REV = 256
WINDOW_REVS = 8
STEP_REVS = 1


def check_pulses_within_vibration(pulse_time: np.ndarray, vib_time: np.ndarray) -> None:
    if pulse_time[0] < vib_time[0] or pulse_time[-1] > vib_time[-1]:
        raise AnalysisError(
            "pulse times must lie within the vibration record "
            f"[{vib_time[0]:.6g}, {vib_time[-1]:.6g}] s"
        )


def window_time_ranges(pulse_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(t_start, t_end) per 8-rev window, stepped by 1 rev. No tail padding."""
    n_windows = pulse_time.size - WINDOW_REVS
    if n_windows < 1:
        raise AnalysisError("fewer than one full 8-revolution window in pulse record")
    starts = pulse_time[:-WINDOW_REVS]
    ends = pulse_time[WINDOW_REVS:]
    return starts, ends


def resample_window(
    vib_time: np.ndarray,
    accel: np.ndarray,
    pulse_time: np.ndarray,
    start_rev: int,
) -> np.ndarray:
    """Linearly interpolate accel at POINTS_PER_REV*WINDOW_REVS equal-angle
    instants spanning revolutions [start_rev, start_rev + WINDOW_REVS).

    Within each pulse interval the shaft angle grows linearly, so the time of
    angle (k + frac) revolutions is t_k + frac * (t_{k+1} - t_k).
    """
    n = POINTS_PER_REV * WINDOW_REVS
    angle = start_rev + np.arange(n) / POINTS_PER_REV
    seg = np.minimum(angle.astype(np.int64), pulse_time.size - 2)
    frac = angle - seg
    t0 = pulse_time[seg]
    t1 = pulse_time[seg + 1]
    sample_times = t0 + frac * (t1 - t0)
    return np.interp(sample_times, vib_time, accel)
