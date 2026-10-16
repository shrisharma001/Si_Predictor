"""make_sample_request.py - writes examples/sample_request.json from the dataset: 72 hourly sensor rows ending at a
test-period cast, the 10 previous Si results, and the query time. Used by the API console and by test_api.py.

  python make_sample_request.py
"""
import json
import os

import numpy as np
import pandas as pd

from run_experiments import DATA_FILE, TEST_FRACTION
from run_extra import load_universe
from si_pipeline import Config

cfg = Config()
sensor = pd.read_excel(DATA_FILE, sheet_name="data")
si = pd.read_excel(DATA_FILE, sheet_name="Si")
sensor["dt"] = pd.to_datetime(sensor["dt"]); si["dt"] = pd.to_datetime(si["dt"])
_, raw, sid, samples = load_universe()
n_dev = int(len(samples) * (1 - TEST_FRACTION))
t = samples.dt.iloc[n_dev + 100]                                     # an ordinary cast in the test period
win = sensor[(sensor.dt <= t) & (sensor.dt > t - pd.Timedelta(hours=72))]
hist = si[si.dt < t].tail(10)
actual = float(samples.loc[samples.dt == t, "Si"].iloc[0])

def rec(df, cols):
    out = []
    for _, r in df.iterrows():
        d = {"dt": r["dt"].isoformat()}
        for c in cols:
            v = pd.to_numeric(r[c], errors="coerce")
            d[c] = None if not np.isfinite(v) else round(float(v), 4)
        out.append(d)
    return out

body = {"query_time": t.isoformat(), "sensor_readings": rec(win, cfg.sensors), "si_history": rec(hist, ["Si"]),
        "_comment": f"Example built from the dataset; the measured Si at the query time was {actual}. Unknown keys are ignored."}
os.makedirs("examples", exist_ok=True)
json.dump(body, open("examples/sample_request.json", "w"), indent=1)
print("wrote examples/sample_request.json:", len(body["sensor_readings"]), "sensor rows,", len(body["si_history"]), "history rows, query", t, "actual Si", actual)
