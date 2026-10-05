"""FastAPI delivery layer: HTTP upload, JSON spectra, CSV download."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .analysis import parse_target_orders, run_analysis, to_csv_bytes, to_json_payload
from .errors import AnalysisError

app = FastAPI(title="rotor_order211", version="0.1.0")


async def _run(
    vibration_csv: UploadFile, pulse_csv: UploadFile, orders: str
):
    try:
        targets = parse_target_orders(orders)
        result = run_analysis(
            await vibration_csv.read(), await pulse_csv.read(), targets
        )
    except AnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@app.post("/analyze")
async def analyze(
    vibration_csv: UploadFile = File(...),
    pulse_csv: UploadFile = File(...),
    orders: str = Form(...),
):
    """Full per-window order spectra as JSON."""
    result = await _run(vibration_csv, pulse_csv, orders)
    return to_json_payload(result)


@app.post("/analyze/csv")
async def analyze_csv(
    vibration_csv: UploadFile = File(...),
    pulse_csv: UploadFile = File(...),
    orders: str = Form(...),
):
    """Target-order amplitudes per window as a CSV download."""
    result = await _run(vibration_csv, pulse_csv, orders)
    return Response(
        content=to_csv_bytes(result),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="target_orders.csv"'},
    )
