"""Orchestrates parsing, angle mapping, spectra and delivery payloads."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np

from .errors import AnalysisError
from .mapping import (
    WINDOW_REVS,
    check_pulses_within_vibration,
    resample_window,
    window_time_ranges,
)
from .parsing import parse_pulses, parse_vibration
from .spectral import order_spectrum

MIN_ORDER = 1
MAX_ORDER = 32



@dataclass
class WindowResult:
    index: int
    rev_start: int
    rev_end: int
    t_start: float
    t_end: float
    t_mid: float
    rpm_mean: float
    orders: np.ndarray
    amplitudes: np.ndarray


@dataclass
class AnalysisResult:
    fs_hz: float
    n_windows: int
    target_orders: list[int]
    windows: list[WindowResult]
    target_amplitudes: np.ndarray  # shape (n_windows, len(target_orders))


def parse_target_orders(raw: str) -> list[int]:
    """Parse a comma-separated list of unique integer orders in [1, 32]."""
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        raise AnalysisError("target orders must be a non-empty list")
    orders: list[int] = []
    for tok in tokens:
        try:
            value = int(tok)
        except ValueError as exc:
            raise AnalysisError(f"target order '{tok}' is not an integer") from exc
        if not (MIN_ORDER <= value <= MAX_ORDER):
            raise AnalysisError(
                f"target order {value} out of range [{MIN_ORDER}, {MAX_ORDER}]"
            )
        if value in orders:
            raise AnalysisError(f"duplicate target order {value}")
        orders.append(value)
    return orders


def _check_nyquist(pulse_time: np.ndarray, fs_hz: float, max_order: int) -> None:
    rot_freq = 1.0 / np.diff(pulse_time)
    worst = max_order * float(rot_freq.max())
    if not worst < fs_hz / 2.0:
        raise AnalysisError(
            f"highest target order x rotation frequency ({worst:.6g} Hz) must be "
            f"below half the vibration sample rate ({fs_hz / 2.0:.6g} Hz)"
        )


def run_analysis(
    vibration_csv: bytes, pulse_csv: bytes, target_orders: list[int]
) -> AnalysisResult:
    vib_time, accel, dt = parse_vibration(vibration_csv)
    pulse_time = parse_pulses(pulse_csv)
    check_pulses_within_vibration(pulse_time, vib_time)
    fs_hz = 1.0 / dt
    _check_nyquist(pulse_time, fs_hz, max(target_orders))

    starts, ends = window_time_ranges(pulse_time)
    n_windows = starts.size
    windows: list[WindowResult] = []
    target_cols: list[np.ndarray] = []
    target_bins = [order * WINDOW_REVS for order in target_orders]

    for w in range(n_windows):
        samples = resample_window(vib_time, accel, pulse_time, w)
        orders, amplitude = order_spectrum(samples)
        t_start = float(starts[w])
        t_end = float(ends[w])
        duration = t_end - t_start
        rpm_mean = 60.0 * WINDOW_REVS / duration  # 480 / window seconds
        windows.append(
            WindowResult(
                index=w,
                rev_start=w,
                rev_end=w + WINDOW_REVS,
                t_start=t_start,
                t_end=t_end,
                t_mid=0.5 * (t_start + t_end),
                rpm_mean=rpm_mean,
                orders=orders,
                amplitudes=amplitude,
            )
        )
        target_cols.append(amplitude[np.asarray(target_bins)])

    return AnalysisResult(
        fs_hz=fs_hz,
        n_windows=n_windows,
        target_orders=target_orders,
        windows=windows,
        target_amplitudes=np.vstack(target_cols),
    )


def to_json_payload(result: AnalysisResult) -> dict:
    return {
        "units": {
            "time": "s",
            "accel_amplitude": "m/s^2 (peak)",
            "rpm": "rev/min",
            "order": "dimensionless (cycles per revolution)",
        },
        "fs_hz": result.fs_hz,
        "window_revs": WINDOW_REVS,
        "target_orders": result.target_orders,
        "n_windows": result.n_windows,
        "windows": [
            {
                "index": w.index,
                "rev_range": [w.rev_start, w.rev_end],
                "t_start_s": w.t_start,
                "t_end_s": w.t_end,
                "t_mid_s": w.t_mid,
                "rpm_mean": w.rpm_mean,
                "orders": w.orders.tolist(),
                "amplitudes": w.amplitudes.tolist(),
            }
            for w in result.windows
        ],
    }


def to_csv_bytes(result: AnalysisResult) -> bytes:
    """Target-order CSV: one row per window, one column per target order."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    header = [
        "window_index",
        "rev_start",
        "rev_end",
        "t_start_s",
        "t_end_s",
        "t_mid_s",
        "rpm_mean",
    ] + [f"order_{o}_amplitude_m_per_s2" for o in result.target_orders]
    writer.writerow(header)
    for w, row in zip(result.windows, result.target_amplitudes):
        writer.writerow(
            [w.index, w.rev_start, w.rev_end, w.t_start, w.t_end, w.t_mid, w.rpm_mean]
            + [float(v) for v in row]
        )
    return buf.getvalue().encode("utf-8")
