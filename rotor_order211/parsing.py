"""CSV parsing and validation for vibration and pulse uploads."""

from __future__ import annotations

import csv
import io

import numpy as np

from .errors import AnalysisError

MAX_VIBRATION_POINTS = 1_000_000
MAX_PULSES = 100_000
MIN_PULSES = 9
DT_REL_TOL = 1e-6


def _read_table(raw: bytes, required: tuple[str, ...], label: str) -> dict[str, list[float]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnalysisError(f"{label}: file is not valid UTF-8") from exc
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        raise AnalysisError(f"{label}: empty CSV")
    header = [cell.strip() for cell in rows[0]]
    missing = [name for name in required if name not in header]
    if missing:
        raise AnalysisError(f"{label}: missing column(s) {missing}")
    columns: dict[str, list[float]] = {name: [] for name in required}
    index = {name: header.index(name) for name in required}
    for lineno, row in enumerate(rows[1:], start=2):
        for name in required:
            try:
                value = float(row[index[name]])
            except (ValueError, IndexError) as exc:
                raise AnalysisError(
                    f"{label}: row {lineno} column '{name}' is not a number"
                ) from exc
            if not np.isfinite(value):
                raise AnalysisError(
                    f"{label}: row {lineno} column '{name}' is not finite"
                )
            columns[name].append(value)
    return columns


def _check_strictly_increasing(time: np.ndarray, label: str) -> None:
    if time.size < 2:
        raise AnalysisError(f"{label}: need at least 2 samples")
    if np.any(np.diff(time) <= 0.0):
        raise AnalysisError(
            f"{label}: time_s must be strictly increasing (no duplicates or reversals)"
        )


def parse_vibration(raw: bytes) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (time_s, accel, dt). Enforces uniform sampling within 1e-6."""
    cols = _read_table(raw, ("time_s", "accel"), "vibration CSV")
    time = np.asarray(cols["time_s"], dtype=np.float64)
    accel = np.asarray(cols["accel"], dtype=np.float64)
    if time.size > MAX_VIBRATION_POINTS:
        raise AnalysisError(f"vibration CSV: exceeds {MAX_VIBRATION_POINTS} points")
    _check_strictly_increasing(time, "vibration CSV")
    dt = np.diff(time)
    dt_mean = float(dt.mean())
    if dt_mean <= 0.0:
        raise AnalysisError("vibration CSV: non-positive sample interval")
    rel_dev = float(np.max(np.abs(dt - dt_mean)) / dt_mean)
    if rel_dev > DT_REL_TOL:
        raise AnalysisError(
            "vibration CSV: sample interval not uniform "
            f"(relative deviation {rel_dev:.3e} > 1e-6)"
        )
    return time, accel, dt_mean


def parse_pulses(raw: bytes) -> np.ndarray:
    """Return pulse times (one pulse per revolution, shared clock)."""
    cols = _read_table(raw, ("time_s",), "pulse CSV")
    time = np.asarray(cols["time_s"], dtype=np.float64)
    if time.size > MAX_PULSES:
        raise AnalysisError(f"pulse CSV: exceeds {MAX_PULSES} pulses")
    if time.size < MIN_PULSES:
        raise AnalysisError(
            f"pulse CSV: need at least {MIN_PULSES} pulses (8-revolution window)"
        )
    _check_strictly_increasing(time, "pulse CSV")
    return time
