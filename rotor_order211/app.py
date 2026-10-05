"""FastAPI application exposing the order-analysis pipeline."""

from __future__ import annotations

import csv
import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .anglemap import WINDOW_REVS, build_windows, max_rotation_freq
from .parsing import (
    DataValidationError,
    parse_pulse_csv,
    parse_vibration_csv,
)
from .spectrum import extract_orders, order_spectrum

app = FastAPI(title="rotor_order211", version="0.1.0")


def _parse_orders(raw: str) -> list[int]:
    try:
        orders = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise HTTPException(422, "orders must be comma-separated integers") from exc
    if not orders:
        raise HTTPException(422, "orders must be non-empty")
    if len(set(orders)) != len(orders):
        raise HTTPException(422, "orders must not contain duplicates")
    if any(o < 1 or o > 32 for o in orders):
        raise HTTPException(422, "orders must be integers in [1, 32]")
    return orders


async def _run_pipeline(
    vibration_file: UploadFile, pulse_file: UploadFile, orders: list[int]
):
    try:
        vib = parse_vibration_csv(await vibration_file.read())
        pulse_times = parse_pulse_csv(await pulse_file.read())
    except DataValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    nyquist = vib.fs / 2.0
    f_rot_max = max_rotation_freq(pulse_times)
    if max(orders) * f_rot_max >= nyquist:
        raise HTTPException(
            422,
            f"highest target order {max(orders)} x max rotation frequency "
            f"{f_rot_max:.6g} Hz must be below half the sampling rate "
            f"({nyquist:.6g} Hz)",
        )

    try:
        windows = build_windows(vib, pulse_times)
    except DataValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    results = []
    for window in windows:
        spectrum = order_spectrum(window)
        targets = extract_orders(spectrum, orders)
        results.append(
            {
                "window": window,
                "spectrum": spectrum,
                "targets": targets,
            }
        )
    return vib, f_rot_max, results


@app.post("/analyze")
async def analyze(
    vibration_file: UploadFile = File(...),
    pulse_file: UploadFile = File(...),
    orders: str = Form(...),
):
    """Full order spectrum per window as JSON."""
    target_orders = _parse_orders(orders)
    vib, f_rot_max, results = await _run_pipeline(
        vibration_file, pulse_file, target_orders
    )
    return {
        "units": {
            "time": "s",
            "accel": "m/s^2",
            "amplitude": "m/s^2 (peak)",
            "rpm": "rev/min",
        },
        "sampling_rate_hz": vib.fs,
        "max_rotation_freq_hz": f_rot_max,
        "points_per_rev": 256,
        "window_revs": WINDOW_REVS,
        "step_revs": 1,
        "target_orders": target_orders,
        "n_windows": len(results),
        "windows": [
            {
                "window_index": i,
                "rev_start": r["window"].rev_start,
                "rev_end": r["window"].rev_start + WINDOW_REVS,
                "time_start_s": r["window"].time_start,
                "time_end_s": r["window"].time_end,
                "time_mid_s": r["window"].time_mid,
                "avg_rpm": r["window"].avg_rpm,
                "orders": r["spectrum"].orders.tolist(),
                "amplitudes": r["spectrum"].amplitudes.tolist(),
                "target_amplitudes": {
                    str(k): v for k, v in r["targets"].items()
                },
            }
            for i, r in enumerate(results)
        ],
    }


@app.post("/analyze/csv")
async def analyze_csv(
    vibration_file: UploadFile = File(...),
    pulse_file: UploadFile = File(...),
    orders: str = Form(...),
):
    """Target-order amplitudes per window as a CSV download."""
    target_orders = _parse_orders(orders)
    _, _, results = await _run_pipeline(vibration_file, pulse_file, target_orders)

    buf = io.StringIO()
    writer = csv.writer(buf)
    header = [
        "window_index",
        "rev_start",
        "rev_end",
        "time_start_s",
        "time_end_s",
        "time_mid_s",
        "avg_rpm",
    ] + [f"order_{o}_amplitude_m_per_s2" for o in target_orders]
    writer.writerow(header)
    for i, r in enumerate(results):
        w = r["window"]
        writer.writerow(
            [
                i,
                w.rev_start,
                w.rev_start + WINDOW_REVS,
                f"{w.time_start:.9f}",
                f"{w.time_end:.9f}",
                f"{w.time_mid:.9f}",
                f"{w.avg_rpm:.6f}",
            ]
            + [f"{r['targets'][o]:.9e}" for o in target_orders]
        )
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=order_analysis.csv"},
    )
