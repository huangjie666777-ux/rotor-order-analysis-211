"""Generate a run-up sample with known order content.

Speed ramps linearly from 10 Hz (600 RPM) to 30 Hz (1800 RPM) over 20 s.
Signal = 1.0*order1 + 0.5*order3 + 0.2*order6 (peak amplitudes, m/s^2).
Run: .venv/bin/python samples/generate_sample.py
"""

import numpy as np

FS = 4096.0
DURATION = 20.0
F0, F1 = 10.0, 30.0  # rotation frequency ramp, Hz


def main() -> None:
    t = np.arange(0.0, DURATION, 1.0 / FS)
    # shaft angle in revolutions: integral of linearly ramping frequency
    angle = F0 * t + 0.5 * (F1 - F0) / DURATION * t**2
    accel = (
        1.0 * np.sin(2 * np.pi * 1 * angle)
        + 0.5 * np.sin(2 * np.pi * 3 * angle)
        + 0.2 * np.sin(2 * np.pi * 6 * angle)
    )
    n_revs = int(angle[-1])
    # pulse times: solve angle(t) = k for k = 0..n_revs
    a = 0.5 * (F1 - F0) / DURATION
    k = np.arange(n_revs + 1)
    pulse_t = (-F0 + np.sqrt(F0**2 + 4 * a * k)) / (2 * a)

    with open("samples/vibration.csv", "w") as fh:
        fh.write("time_s,accel\n")
        for ti, ai in zip(t, accel):
            fh.write(f"{ti:.12f},{ai:.9f}\n")
    with open("samples/pulses.csv", "w") as fh:
        fh.write("time_s\n")
        for ti in pulse_t:
            fh.write(f"{ti:.12f}\n")
    print(f"wrote {t.size} vibration samples, {pulse_t.size} pulses")


if __name__ == "__main__":
    main()
