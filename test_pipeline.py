"""
test_pipeline.py - checks that the deployment pipeline reproduces the experiments and behaves safely.

  python test_pipeline.py            # needs data/1_blast_furnace_data_first_dataset.xlsx and results/intervals_test_results.csv
Exits with a non-zero status if a check fails.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

from run_experiments import CV_BOUNDS, DATA_FILE, TEST_FRACTION
from run_extra import load_universe
try:
    from si_pipeline import require_version
except ImportError:
    raise SystemExit("Your si_pipeline.py is out of date: copy the latest si_pipeline.py from the project folder over yours.")
require_version("1.3")
from si_pipeline import SiPipeline, evaluate

FAILED = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not cond:
        FAILED.append(name)


sensor = pd.read_excel(DATA_FILE, sheet_name="data")
si_raw = pd.read_excel(DATA_FILE, sheet_name="Si")
sensor["dt"] = pd.to_datetime(sensor["dt"]); si_raw["dt"] = pd.to_datetime(si_raw["dt"])
_, raw, si, samples = load_universe()
n = len(samples); n_dev = int(n * (1 - TEST_FRACTION))
dev, test = samples.iloc[:n_dev], samples.iloc[n_dev:]
dev_end = dev.dt.iloc[-1]

# ---- 1. dev-only fit must reproduce the experiment results on the test period ----------------------------------
print("\n[1] reproduce the experiments (fit on the development period only)")
pipe = SiPipeline(use_lags=False, use_roll=True, use_history=True, model="hybrid", calibrate=True, refit_on_all=False)
pipe.fit(sensor[sensor.dt <= dev_end], si_raw[si_raw.dt <= dev_end])
out = pipe.predict_frame(sensor, test.dt.values, si_history=si_raw)
y = test.set_index("dt").loc[pd.DatetimeIndex(out.time), "Si"].values
m = evaluate(y, out.prediction.values)
cov = float(((y >= out.lower) & (y <= out.upper)).mean()); width = float((out.upper - out.lower).mean())
ref = pd.read_csv("results/intervals_test_results.csv").set_index("method").loc["conformal"]
print(f"      pipeline: MAE={m['MAE']:.4f} R2={m['R2']:.3f} coverage={cov:.3f} width={width:.3f}")
print(f"      experiments: coverage={ref.coverage:.3f} width={ref.width:.3f}; hybrid test MAE 0.0812, R2 0.722 (consolidated table)")
check("test MAE within 0.003 of the experiments (0.0812)", abs(m["MAE"] - 0.0812) < 0.003, f"{m['MAE']:.4f}")
check("test coverage within 0.02 of the experiments", abs(cov - ref.coverage) < 0.02, f"{cov:.3f} vs {ref.coverage:.3f}")
check("interval width within 0.03 of the experiments", abs(width - ref.width) < 0.03, f"{width:.3f} vs {ref.width:.3f}")
check("every test query got a prediction", out.prediction.notna().all(), f"{out.prediction.notna().sum()}/{len(out)}")

# ---- 2. input validation and warnings -------------------------------------------------------------------------
print("\n[2] input validation")
t0 = test.dt.iloc[100]
window = sensor[(sensor.dt <= t0) & (sensor.dt > t0 - pd.Timedelta(hours=60))].copy()
hist = si_raw[si_raw.dt < t0]
base = pipe.predict_frame(window, [t0], si_history=hist)
check("a normal request works with 60 h of sensor data", base.status.iloc[0] in ("ok", "warning") and np.isfinite(base.prediction.iloc[0]),
      f"{base.prediction.iloc[0]:.3f} ({base.status.iloc[0]})")
try:
    pipe.predict_frame(window.drop(columns=["Tp4"]), [t0], si_history=hist); ok = False
except ValueError as e:
    ok = "Tp4" in str(e)
check("missing sensor column raises a clear error", ok)
try:
    pipe.predict_frame(window, [t0], si_history=None); ok = False
except ValueError:
    ok = True
check("missing Si history raises a clear error", ok)
short = pipe.predict_frame(window[window.dt > t0 - pd.Timedelta(hours=5)], [t0], si_history=hist)
check("too little sensor history gives no_prediction with a reason", short.status.iloc[0] == "no_prediction" and "history" in short.messages.iloc[0], short.messages.iloc[0])
few = pipe.predict_frame(window, [t0], si_history=si_raw[si_raw.dt < si_raw.dt.min() + pd.Timedelta(hours=3)])
check("fewer than 3 previous Si results gives no_prediction", few.status.iloc[0] == "no_prediction", few.messages.iloc[0])
old_t = t0 + pd.Timedelta(hours=0)  # stale previous result: only results older than 9 h
stale = pipe.predict_frame(window, [t0], si_history=si_raw[si_raw.dt < t0 - pd.Timedelta(hours=9)])
check("stale previous Si result triggers a warning", "old" in stale.messages.iloc[0], stale.messages.iloc[0][:80])
bad = window.copy(); bad.loc[bad.index[-1], "Tp4"] = 3311.2
fb = pipe.predict_frame(bad, [t0], si_history=hist)
check("a fault code (3311.2 in Tp4) is flagged and the prediction stays close",
      "Tp4" in fb.messages.iloc[0] and abs(fb.prediction.iloc[0] - base.prediction.iloc[0]) < 0.1,
      f"{fb.prediction.iloc[0]:.3f} vs {base.prediction.iloc[0]:.3f}")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "p.pkl"); pipe.save(p)
    again = SiPipeline.load(p).predict_frame(window, [t0], si_history=hist)
check("save and load give identical predictions", np.allclose(again.prediction, base.prediction))

# ---- 3. coverage with and without refitting on all data (development period only) ------------------------------
print("\n[3] interval coverage, refit_on_all False vs True (5 expanding-window folds, development period only)")
bounds = [int(n_dev * b) for b in CV_BOUNDS]
res = {False: [], True: []}
for k in range(len(bounds) - 1):
    a, b = bounds[k], bounds[k + 1]
    end = dev.dt.iloc[a - 1]
    val = dev.iloc[a:b]
    for refit in (False, True):
        pp = SiPipeline(use_lags=False, use_roll=True, use_history=True, model="hybrid", calibrate=True, refit_on_all=refit)
        pp.fit(sensor[sensor.dt <= end], si_raw[si_raw.dt <= end])
        o = pp.predict_frame(sensor, val.dt.values, si_history=si_raw)
        yy = val.set_index("dt").loc[pd.DatetimeIndex(o.time), "Si"].values
        res[refit].append((float(((yy >= o.lower) & (yy <= o.upper)).mean()), float((o.upper - o.lower).mean()),
                           float(np.abs(yy - o.prediction.values).mean())))
    print(f"      fold {k+1}: no refit cov={res[False][-1][0]:.3f} w={res[False][-1][1]:.3f} mae={res[False][-1][2]:.4f} | "
          f"refit cov={res[True][-1][0]:.3f} w={res[True][-1][1]:.3f} mae={res[True][-1][2]:.4f}", flush=True)
for refit in (False, True):
    arr = np.array(res[refit])
    print(f"      refit_on_all={refit}: mean coverage {arr[:,0].mean():.3f}, width {arr[:,1].mean():.3f}, MAE {arr[:,2].mean():.4f}")
check("mean CV coverage close to the 90 % target (0.85-0.93) with refit_on_all=True", 0.85 <= np.array(res[True])[:, 0].mean() <= 0.93,
      f"{np.array(res[True])[:,0].mean():.3f}")

print("\nRESULT:", "ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} CHECK(S) FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
