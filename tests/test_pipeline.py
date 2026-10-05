"""End-to-end and validation tests for rotor_order211."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rotor_order211.app import app
from rotor_order211.anglemap import WINDOW_REVS, AngleWindow, build_windows
from rotor_order211.parsing import (
    DataValidationError,
    parse_pulse_csv,
    parse_vibration_csv,
)
from rotor_order211.spectrum import extract_orders, order_spectrum, periodic_hann

FS = 4096.0
DURATION_S = 4.0


def make_run(orders_amp, rpm0=600.0, rpm1=1500.0, duration=DURATION_S, fs=FS):
    n = int(fs * duration) + 1
    t = np.arange(n) / fs
    f0, f1 = rpm0 / 60.0, rpm1 / 60.0
    k = (f1 - f0) / duration
    revs = f0 * t + 0.5 * k * t**2
    phase = 2.0 * np.pi * revs
    accel = np.zeros_like(t)
    for order, amp in orders_amp.items():
        accel += amp * np.sin(order * phase)
    n_revs = int(revs[-1])
    pulse_times = np.interp(np.arange(n_revs + 1), revs, t)
    return t, accel, pulse_times


def to_csv(rows, header):
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for row in rows:
        buf.write(",".join(str(v) for v in row) + "\n")
    return buf.getvalue().encode()


def vib_csv(t, accel):
    return to_csv(zip(t, accel), ["time_s", "accel"])


def pulse_csv(pt):
    return to_csv(((p,) for p in pt), ["time_s"])


def test_known_orders_recovered():
    amps = {1: 0.5, 2: 0.3, 5: 0.2}
    t, accel, pt = make_run(amps)
    vib = parse_vibration_csv(vib_csv(t, accel))
    pulses = parse_pulse_csv(pulse_csv(pt))
    windows = build_windows(vib, pulses)
    assert len(windows) == len(pt) - 1 - WINDOW_REVS + 1
    spec = order_spectrum(windows[len(windows) // 2])
    got = extract_orders(spec, sorted(amps))
    for order, amp in amps.items():
        assert got[order] == pytest.approx(amp, rel=0.02)


def test_avg_rpm_and_mid_time():
    t, accel, pt = make_run({1: 0.1})
    vib = parse_vibration_csv(vib_csv(t, accel))
    pulses = parse_pulse_csv(pulse_csv(pt))
    w = build_windows(vib, pulses)[0]
    assert w.time_mid == pytest.approx(0.5 * (w.time_start + w.time_end))
    assert w.avg_rpm == pytest.approx(480.0 / (w.time_end - w.time_start))


def test_nyquist_not_doubled():
    n = WINDOW_REVS * 256
    w = periodic_hann(n)
    nyq = (-1.0) ** np.arange(n)
    win = AngleWindow(0, 0.0, 1.0, 0.5, 480.0, nyq)
    spec = order_spectrum(win)
    # Nyquist peak amplitude is preserved without doubling.
    assert spec.amplitudes[-1] == pytest.approx(1.0, rel=1e-12)


def test_reject_missing_column():
    with pytest.raises(DataValidationError):
        parse_vibration_csv(b"time_s,vel\n0.0,1.0\n0.1,2.0\n")


def test_reject_non_finite():
    with pytest.raises(DataValidationError):
        parse_vibration_csv(b"time_s,accel\n0.0,nan\n0.1,1.0\n")


def test_reject_reversed_time():
    with pytest.raises(DataValidationError):
        parse_pulse_csv(b"time_s\n0.2\n0.1\n")


def test_reject_duplicate_time():
    with pytest.raises(DataValidationError):
        parse_pulse_csv(b"time_s\n0.1\n0.1\n")


def test_reject_nonuniform_dt():
    t = np.array([0.0, 0.1, 0.21])
    with pytest.raises(DataValidationError):
        parse_vibration_csv(vib_csv(t, np.zeros(3)))


def test_reject_too_few_pulses():
    with pytest.raises(DataValidationError):
        parse_pulse_csv(pulse_csv(np.arange(8) * 0.1))


def test_reject_pulses_outside_record():
    t, accel, pt = make_run({1: 0.1})
    vib = parse_vibration_csv(vib_csv(t, accel))
    with pytest.raises(DataValidationError):
        build_windows(vib, pt + 1000.0)


def upload(c, vib_bytes, pulse_bytes, orders="1,2,5", path="/analyze"):
    return c.post(
        path,
        files={
            "vibration_file": ("vibration.csv", vib_bytes, "text/csv"),
            "pulse_file": ("pulses.csv", pulse_bytes, "text/csv"),
        },
        data={"orders": orders},
    )


def test_api_json_and_csv():
    t, accel, pt = make_run({1: 0.5, 2: 0.3, 5: 0.2})
    vb, pb = vib_csv(t, accel), pulse_csv(pt)
    c = TestClient(app)
    resp = upload(c, vb, pb)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_windows"] == len(pt) - 1 - WINDOW_REVS + 1
    w0 = body["windows"][0]
    assert len(w0["orders"]) == WINDOW_REVS * 256 // 2 + 1
    assert w0["target_amplitudes"]["1"] == pytest.approx(0.5, rel=0.05)
    assert body["units"]["accel"] == "m/s^2"

    resp_csv = upload(c, vb, pb, path="/analyze/csv")
    assert resp_csv.status_code == 200
    assert "order_5_amplitude_m_per_s2" in resp_csv.text.splitlines()[0]


def test_api_rejects_bad_orders():
    t, accel, pt = make_run({1: 0.1})
    vb, pb = vib_csv(t, accel), pulse_csv(pt)
    c = TestClient(app)
    assert upload(c, vb, pb, "1,1").status_code == 422
    assert upload(c, vb, pb, "0").status_code == 422
    assert upload(c, vb, pb, "33").status_code == 422
    assert upload(c, vb, pb, "").status_code == 422


def test_api_rejects_nyquist_violation():
    t, accel, pt = make_run({1: 0.1}, fs=1000.0)
    vb, pb = vib_csv(t, accel), pulse_csv(pt)
    c = TestClient(app)
    assert upload(c, vb, pb, "32").status_code == 422
    assert upload(c, vb, pb, "1").status_code == 200
