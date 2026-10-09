"""
run_experiments.py
==================
Evaluation protocol (fixed BEFORE looking at results, so it can be described in the paper)

* All aligned Si samples are sorted by time.
* The LAST 15 % is the TEST set. It is never used for model/feature selection.
  It is evaluated once, by `--stage final`, for configurations chosen from CV.
* The first 85 % ("dev") is used for expanding-window time-series
  cross-validation (5 folds, each validating on the next ~12 % block).
* Every fold re-fits the sensor cleaner on that fold's training period only.
* Every model sees exactly the same rows, so numbers are directly comparable.

Usage
-----
  python run_experiments.py --stage cv
  python run_experiments.py --stage final --configs "F3+H:xgb_huber,F0:xgb_l2,F1:xgb_l2"
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from si_pipeline import (Config, SensorCleaner, assemble, build_sensor_features, evaluate,
                         make_model, prepare_sensor_frame, prepare_si_frame, valid_samples)

DATA_FILE = "data/1_blast_furnace_data_first_dataset.xlsx"
OUT_DIR = "results"
TEST_FRACTION = 0.15
CV_BOUNDS = [0.40, 0.52, 0.64, 0.76, 0.88, 1.00]    # fractions of the dev set

# name -> (use_lags, use_roll)
FEATURE_SETS = {
    "F0": (False, False),   # 27 raw sensors + blast_off flag
    "F1": (True, False),    # + selected lags (from the existing lag analysis)
    "F2": (False, True),    # + rolling mean/std
    "F3": (True, True),     # + both
}
CV_MODELS_NO_HIST = ["mean", "ridge", "rf", "xgb_l2", "xgb_huber"]
CV_MODELS_HIST = ["persistence", "ridge", "rf", "xgb_l2", "xgb_huber"]


def parse_cfg(name):
    """'F3+H' -> ('F3', use_history=True)"""
    hist = name.endswith("+H")
    return name.replace("+H", ""), hist


def load():
    cfg = Config()
    sensor = pd.read_excel(DATA_FILE, sheet_name="data")
    si = pd.read_excel(DATA_FILE, sheet_name="Si")
    raw = prepare_sensor_frame(sensor, cfg)
    si = prepare_si_frame(si, cfg)
    samples = valid_samples(si, raw.index, cfg)
    return cfg, raw, si, samples


def datasets_for(clean, raw, samples, si, cfg):
    """All feature-set variants for one cleaner state."""
    out = {}
    for fs, (lags, roll) in FEATURE_SETS.items():
        feats = build_sensor_features(clean, raw, cfg, lags, roll)
        for hist in (False, True):
            name = fs + ("+H" if hist else "")
            out[name] = assemble(feats, samples, si, cfg, hist)
    return out


def split_cols(d, cfg):
    return [c for c in d.columns if c not in (cfg.time_col, cfg.target)]


def run_cv(args):
    cfg, raw, si, samples = load()
    n = len(samples)
    n_test_start = int(n * (1 - TEST_FRACTION))
    dev = samples.iloc[:n_test_start]
    n_dev = len(dev)
    bounds = [int(n_dev * b) for b in CV_BOUNDS]
    print(f"[INFO] samples={n}  dev={n_dev}  test={n - n_test_start}")
    print(f"[INFO] dev {dev.dt.min()} -> {dev.dt.max()}")
    print(f"[INFO] test starts {samples.dt.iloc[n_test_start]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    path = os.path.join(OUT_DIR, "cv_fold_results.csv")
    t_start = time.time()
    for k in range(len(bounds) - 1):
        tr_end, va_end = bounds[k], bounds[k + 1]
        train_end_time = dev.dt.iloc[tr_end - 1]
        cleaner = SensorCleaner().fit(raw.loc[:train_end_time], cfg.sensors)
        clean = cleaner.transform(raw)
        data = datasets_for(clean, raw, samples.iloc[:n_dev], si, cfg)
        for name, d in data.items():
            assert len(d) == n_dev, (name, len(d))
            cols = split_cols(d, cfg)
            tr, va = d.iloc[:tr_end], d.iloc[tr_end:va_end]
            models = CV_MODELS_HIST if name.endswith("+H") else CV_MODELS_NO_HIST
            for mname in models:
                t0 = time.time()
                m = make_model(mname).fit(tr[cols], tr[cfg.target])
                p = m.predict(va[cols])
                r = evaluate(va[cfg.target], p)
                r.update(fold=k + 1, features=name, model=mname,
                         n_features=len(cols), train_rows=len(tr),
                         val_start=str(va.dt.iloc[0].date()), val_end=str(va.dt.iloc[-1].date()),
                         sec=round(time.time() - t0, 1))
                rows.append(r)
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"fold {k+1} {name:6s} {mname:12s} R2={r['R2']:7.3f} MAE={r['MAE']:.4f} "
                      f"in0.1={r['within_0.10']:.3f} ({r['sec']}s, total {(time.time()-t_start)/60:.1f} min)",
                      flush=True)

    df = pd.DataFrame(rows)
    summ = (df.groupby(["features", "model"])[["MAE", "RMSE", "R2", "within_0.10", "R2_normal", "MAE_normal"]]
              .agg(["mean", "std"]).round(4))
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.sort_values("MAE_mean")
    summ.to_csv(os.path.join(OUT_DIR, "cv_summary.csv"))
    print("\n=== CV SUMMARY (sorted by mean MAE) ===")
    print(summ.to_string())


def run_final(args):
    cfg, raw, si, samples = load()
    n = len(samples)
    n_test_start = int(n * (1 - TEST_FRACTION))
    dev_end_time = samples.dt.iloc[n_test_start - 1]
    cleaner = SensorCleaner().fit(raw.loc[:dev_end_time], cfg.sensors)
    print("[INFO] cleaner fitted on dev period only; faults found:")
    print(cleaner.report().to_string())
    clean = cleaner.transform(raw)
    data = datasets_for(clean, raw, samples, si, cfg)
    rows, preds = [], {}
    for item in args.configs.split(","):
        name, mname = item.strip().split(":")
        d = data[name]
        cols = split_cols(d, cfg)
        tr, te = d.iloc[:n_test_start], d.iloc[n_test_start:]
        m = make_model(mname).fit(tr[cols], tr[cfg.target])
        p = m.predict(te[cols])
        r = evaluate(te[cfg.target], p)
        r.update(features=name, model=mname, n_features=len(cols), train_rows=len(tr))
        rows.append(r)
        preds[f"{name}:{mname}"] = p
        print(f"TEST {name:6s} {mname:12s} R2={r['R2']:7.3f} MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} "
              f"in0.1={r['within_0.10']:.3f} R2_normal={r['R2_normal']:.3f}", flush=True)
        if hasattr(m, "feature_importances_"):
            imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
            imp.to_csv(os.path.join(OUT_DIR, f"importance_{name}_{mname}.csv".replace("+", "H")))
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "final_test_results.csv"), index=False)
    out = te[[cfg.time_col, cfg.target]].copy()
    for k, v in preds.items():
        out[k] = v
    out.to_csv(os.path.join(OUT_DIR, "final_test_predictions.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cv", "final"], required=True)
    ap.add_argument("--configs", default="F0:xgb_l2")
    a = ap.parse_args()
    run_cv(a) if a.stage == "cv" else run_final(a)
