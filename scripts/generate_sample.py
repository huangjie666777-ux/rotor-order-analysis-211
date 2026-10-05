"""Generate a variable-speed sample with known orders 1, 2 and 5.

Speed ramps linearly from 600 to 1800 rpm. The vibration signal contains
order components of known peak amplitude plus a little noise, sampled at a
fixed rate; one pulse per revolution is emitted on the same clock.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

FS = 8192.0
DURATION_S = 8.0
RPM_START = 600.0
RPM_END = 1800.0
ORDERS = {1: 0.50, 2: 0.30, 5: 0.20}  # order -> peak amplitude (m/s^2)
NOISE_STD = 0.02

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    rng = np.random.default_rng(42)
    n = int(FS * DURATION_S) + 1
    t = np.arange(n) / FS
    f0 = RPM_START / 60.0
    f1 = RPM_END / 60.0
    k = (f1 - f0) / DURATION_S
    revs = f0 * t + 0.5 * k * t**2
    phase = 2.0 * np.pi * revs

    accel = np.zeros_like(t)
    for order, amp in ORDERS.items():
        accel += amp * np.sin(order * phase + 0.3 * order)
    accel += NOISE_STD * rng.standard_normal(n)

    n_revs = int(revs[-1])
    pulse_times = np.interp(np.arange(n_revs + 1), revs, t)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "vibration.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s", "accel"])
        for ti, ai in zip(t, accel):
            writer.writerow([f"{ti:.12f}", f"{ai:.9e}"])
    with (OUT_DIR / "pulses.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s"])
        for pt in pulse_times:
            writer.writerow([f"{pt:.12f}"])
    print(f"wrote {n} vibration samples and {pulse_times.size} pulses to {OUT_DIR}")
    print(f"revolutions: {n_revs}, expected orders: {ORDERS}")


if __name__ == "__main__":
    main()
