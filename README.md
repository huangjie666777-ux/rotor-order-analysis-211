# rotor_order211

Pure-backend order analysis for rotating machinery during run-up / run-down.
Python 3.10.12, FastAPI 0.115.12, NumPy 2.2.6.

## Layout

- `rotor_order211/parsing.py` — CSV parsing and validation (columns, finite
  values, strictly increasing time, uniform sampling within 1e-6, size caps
  of 1e6 vibration points / 1e5 pulses, at least 9 pulses).
- `rotor_order211/mapping.py` — pulse-to-angle mapping and equal-angle
  resampling (256 points/rev, 8-rev windows, 1-rev step, no extrapolation,
  no tail padding).
- `rotor_order211/spectral.py` — demean, periodic Hann window, real FFT,
  single-sided peak amplitudes normalised by the window weight sum.
- `rotor_order211/analysis.py` — orchestration, Nyquist guard, JSON/CSV
  payload builders.
- `rotor_order211/api.py` — FastAPI endpoints.

## Input

Two UTF-8 CSV uploads sharing one clock:

- `vibration_csv`: columns `time_s` (s, increasing, equally spaced) and
  `accel` (m/s^2).
- `pulse_csv`: column `time_s`, one pulse per full revolution. The first
  pulse is revolution 0; angle grows linearly between pulses. All pulses
  must lie inside the vibration record.

Form field `orders`: comma-separated, non-empty, unique integers in 1..32.

## Method

Each 8-revolution window is resampled to 8*256 equal-angle instants by
linear interpolation of the time-domain signal (angle maps linearly to time
within each pulse interval; the whole-record average speed is never used).
Per window: demean, periodic Hann, rFFT. Bin `i` is order `i/8`. Amplitudes
are divided by the window weight sum; positive frequencies are doubled
except DC and Nyquist, giving single-sided peak amplitude. Per window the
API reports the midpoint time, mean RPM = 480 / window_seconds, and the
full order spectrum; target orders are extracted at bin `order*8`.

The highest target order times every pulse-interval rotation frequency must
be below half the vibration sample rate, otherwise the request is rejected.

## Assumptions and limits

- Linear interpolation between the original samples assumes the vibration
  signal is band-limited well below the Nyquist frequency; the equal-angle
  resampling inherits that assumption.
- Angle is piecewise-linear between pulses; intra-revolution speed
  fluctuation (torsional vibration) is not modelled.
- The service performs signal analysis only; it does not diagnose faults.

## Run

    .venv/bin/python -m rotor_order211          # serves on 127.0.0.1:8211
    .venv/bin/python -m pytest tests -q         # self-test
    .venv/bin/python samples/generate_sample.py # known-order run-up sample

## Demo

    # JSON: full order spectrum per window
    curl -s -F vibration_csv=@samples/vibration.csv -F pulse_csv=@samples/pulses.csv \
         -F orders=1,3,6 http://127.0.0.1:8211/analyze

    # CSV download: target-order amplitudes per window
    curl -s -F vibration_csv=@samples/vibration.csv -F pulse_csv=@samples/pulses.csv \
         -F orders=1,3,6 http://127.0.0.1:8211/analyze/csv -o target_orders.csv

The sample ramps from 600 to 1800 RPM with known peak amplitudes
1.0 / 0.5 / 0.2 m/s^2 at orders 1 / 3 / 6.
