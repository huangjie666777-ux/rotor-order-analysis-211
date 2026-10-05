import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rotor_order211.analysis import parse_target_orders, run_analysis
from rotor_order211.api import app
from rotor_order211.errors import AnalysisError

FS = 4096.0


def make_data(n_pulses=40, f0=10.0, f1=30.0, duration=20.0, amps=(1.0, 0.5, 0.2)):
    t = np.arange(0.0, duration, 1.0 / FS)
    angle = f0 * t + 0.5 * (f1 - f0) / duration * t**2
    accel = sum(a * np.sin(2 * np.pi * o * angle) for o, a in zip((1, 3, 6), amps))
    a = 0.5 * (f1 - f0) / duration
    k = np.arange(n_pulses)
    pulse_t = (-f0 + np.sqrt(f0**2 + 4 * a * k)) / (2 * a)
    vib = "time_s,accel\n" + "\n".join(f"{x:.12f},{y:.9f}" for x, y in zip(t, accel))
    pul = "time_s\n" + "\n".join(f"{x:.9f}" for x in pulse_t)
    return vib.encode(), pul.encode()


def test_known_orders_recovered():
    vib, pul = make_data()
    result = run_analysis(vib, pul, [1, 3, 6])
    assert result.n_windows == 40 - 8
    mid = result.target_amplitudes[result.n_windows // 2]
    assert mid[0] == pytest.approx(1.0, rel=2e-2)
    assert mid[1] == pytest.approx(0.5, rel=2e-2)
    assert mid[2] == pytest.approx(0.2, rel=2e-2)
    rpm = [w.rpm_mean for w in result.windows]
    assert rpm[0] < rpm[-1]  # run-up: RPM increases per window


def test_reject_few_pulses():
    vib, _ = make_data()
    with pytest.raises(AnalysisError):
        run_analysis(vib, b"time_s\n0.1\n0.2\n", [1])


def test_reject_duplicate_times():
    with pytest.raises(AnalysisError):
        run_analysis(b"time_s,accel\n0,0\n0,1\n0.1,2\n", b"time_s\n", [1])


def test_reject_missing_column():
    with pytest.raises(AnalysisError):
        run_analysis(b"time_s\n0\n1\n", b"time_s\n0\n", [1])


def test_reject_nonuniform_dt():
    vib = b"time_s,accel\n0,0\n0.1,0\n0.1002,0\n"
    with pytest.raises(AnalysisError):
        run_analysis(vib, b"time_s\n0\n", [1])


def test_reject_nyquist_violation():
    vib, _ = make_data()
    # pulses every 5 ms -> 200 Hz rotation; 32*200 = 6400 > FS/2 = 2048
    pul2 = ("time_s\n" + "\n".join(f"{0.005 * k:.9f}" for k in range(20))).encode()
    with pytest.raises(AnalysisError, match="half the vibration sample rate"):
        run_analysis(vib, pul2, [32])


def test_target_order_validation():
    assert parse_target_orders("1, 3, 6") == [1, 3, 6]
    for bad in ["", "0", "33", "1,1", "x"]:
        with pytest.raises(AnalysisError):
            parse_target_orders(bad)


def test_api_json_and_csv():
    vib, pul = make_data()
    client = TestClient(app)
    files = {
        "vibration_csv": ("vibration.csv", io.BytesIO(vib), "text/csv"),
        "pulse_csv": ("pulses.csv", io.BytesIO(pul), "text/csv"),
    }
    r = client.post("/analyze", files=files, data={"orders": "1,3,6"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_windows"] == 32
    assert len(body["windows"][0]["orders"]) == 1025
    r2 = client.post("/analyze/csv", files=files, data={"orders": "1,3,6"})
    assert r2.status_code == 200
    assert "order_6_amplitude_m_per_s2" in r2.text.splitlines()[0]
    r3 = client.post("/analyze", files=files, data={"orders": "1,1"})
    assert r3.status_code == 422
