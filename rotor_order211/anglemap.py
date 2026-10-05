"""Pulse-to-angle mapping and constant-angle resampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parsing import DataValidationError, VibrationData

POINTS_PER_REV = 256
WINDOW_REVS = 8
STEP_REVS = 1


@dataclass(frozen=True)
class AngleWindow:
    rev_start: int
    time_start: float
    time_end: float
    time_mid: float
    avg_rpm: float
    samples: np.ndarray


def build_windows(vib: VibrationData, pulse_times: np.ndarray) -> list[AngleWindow]:
    """Resample vibration onto 256 points/rev for each 8-rev window, step 1 rev.

    Angle (in revolutions) grows linearly between consecutive pulses; the
    first pulse is revolution 0. No extrapolation or tail padding.
    """
    t = vib.time_s
    if pulse_times[0] < t[0] or pulse_times[-1] > t[-1]:
        raise DataValidationError(
            "pulse times must lie within the vibration record"
        )
    n_revs = pulse_times.size - 1
    if n_revs < WINDOW_REVS:
        raise DataValidationError(
            f"need at least {WINDOW_REVS + 1} pulses ({WINDOW_REVS} revolutions) "
            f"for one analysis window, got {pulse_times.size} pulses"
        )

    rev_t = pulse_times
    windows: list[AngleWindow] = []
    n_per_window = WINDOW_REVS * POINTS_PER_REV
    rev_index = np.arange(rev_t.size, dtype=np.float64)
    for rev_start in range(0, n_revs - WINDOW_REVS + 1, STEP_REVS):
        rev_end = rev_start + WINDOW_REVS
        t_start = float(rev_t[rev_start])
        t_end = float(rev_t[rev_end])
        frac = np.arange(n_per_window, dtype=np.float64) / POINTS_PER_REV
        rev_pos = rev_start + frac
        sample_times = np.interp(rev_pos, rev_index, rev_t)
        samples = np.interp(sample_times, t, vib.accel)
        duration = t_end - t_start
        windows.append(
            AngleWindow(
                rev_start=rev_start,
                time_start=t_start,
                time_end=t_end,
                time_mid=0.5 * (t_start + t_end),
                avg_rpm=60.0 * WINDOW_REVS / duration,
                samples=samples,
            )
        )
    return windows


def max_rotation_freq(pulse_times: np.ndarray) -> float:
    """Highest per-interval rotational frequency (rev/s)."""
    return float(1.0 / np.min(np.diff(pulse_times)))
