"""CSV parsing and validation for vibration and pulse uploads."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np

MAX_VIBRATION_POINTS = 1_000_000
MAX_PULSES = 100_000
MIN_PULSES = 9
TIME_REL_TOL = 1e-6


class DataValidationError(ValueError):
    """Raised when an uploaded CSV fails validation (HTTP 422)."""


def _read_csv(text):
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        raise DataValidationError("CSV is empty")
    header = [cell.strip() for cell in rows[0]]
    return header, rows[1:]


def _column(header, rows, name):
    if name not in header:
        raise DataValidationError(f"missing required column '{name}'")
    idx = header.index(name)
    values = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        if idx >= len(row):
            raise DataValidationError(f"row {i + 2}: missing value for '{name}'")
        try:
            values[i] = float(row[idx])
        except ValueError as exc:
            raise DataValidationError(
                f"row {i + 2}: '{row[idx]}' is not a number"
            ) from exc
    return values


def _check_time(time_s, label):
    if time_s.size == 0:
        raise DataValidationError(f"{label}: no data rows")
    if not np.all(np.isfinite(time_s)):
        raise DataValidationError(f"{label}: time_s contains non-finite values")
    diffs = np.diff(time_s)
    if np.any(diffs <= 0):
        raise DataValidationError(
            f"{label}: time_s must be strictly increasing (duplicate or reversed time)"
        )


@dataclass(frozen=True)
class VibrationData:
    time_s: np.ndarray
    accel: np.ndarray
    fs: float


def parse_vibration_csv(raw: bytes) -> VibrationData:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataValidationError("vibration CSV must be UTF-8") from exc
    header, rows = _read_csv(text)
    if len(rows) > MAX_VIBRATION_POINTS:
        raise DataValidationError(
            f"vibration CSV exceeds {MAX_VIBRATION_POINTS} points"
        )
    time_s = _column(header, rows, "time_s")
    accel = _column(header, rows, "accel")
    _check_time(time_s, "vibration")
    if not np.all(np.isfinite(accel)):
        raise DataValidationError("vibration: accel contains non-finite values")
    if time_s.size < 2:
        raise DataValidationError("vibration: need at least 2 samples")
    dt = np.diff(time_s)
    mean_dt = float(dt.mean())
    if float(np.max(np.abs(dt - mean_dt))) / mean_dt > TIME_REL_TOL:
        raise DataValidationError(
            "vibration: time_s is not uniformly spaced (rel deviation > 1e-6)"
        )
    return VibrationData(time_s=time_s, accel=accel, fs=1.0 / mean_dt)


def parse_pulse_csv(raw: bytes) -> np.ndarray:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataValidationError("pulse CSV must be UTF-8") from exc
    header, rows = _read_csv(text)
    if len(rows) > MAX_PULSES:
        raise DataValidationError(f"pulse CSV exceeds {MAX_PULSES} pulses")
    time_s = _column(header, rows, "time_s")
    _check_time(time_s, "pulse")
    if time_s.size < MIN_PULSES:
        raise DataValidationError(
            f"pulse: need at least {MIN_PULSES} pulses, got {time_s.size}"
        )
    return time_s
