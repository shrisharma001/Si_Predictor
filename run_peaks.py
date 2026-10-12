"""
run_peaks.py - can we do better on upsets (Si > 1.2) without hurting normal operation?

Variants (all use 3 previous Si results):
  base        : F2+H Ridge (current primary model)
  ew          : F2 + early-warning features (3 h change, deviation from 24 h mean/max/min), Ridge
  log         : base features, Ridge trained on log(Si), back-transformed
  ew_log      : early-warning features + log target
  hybrid      : Ridge, then XGBoost on Ridge residuals (lets trees correct, linear part can extrapolate)
  ew_hybrid   : early-warning features + hybrid
  xgb_ref     : F0+H XGBoost reference

Selection rule fixed in advance: the single new variant with the lowest mean CV MAE is evaluated ONCE
on the test set, next to the reference models.
"""
import os
import time

import numpy as np
import pandas as pd

from run_experiments import CV_BOUNDS, OUT_DIR, TEST_FRACTION, split_cols
from run_extra import load_universe
from si_pipeline import (Config, SensorCleaner, assemble, build_sensor_features, evaluate, make_model)


class LogTarget:
    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        self.base.fit(X, np.log(y))
        return self

    def predict(self, X):
        return np.exp(self.base.predict(X))


class Hybrid:
    """Ridge + XGBoost fitted to the Ridge residuals."""
    def __init__(self):
        self.lin, self.res = make_model("ridge"), make_model("xgb_l2")

    def fit(self, X, y):
        self.lin.fit(X, y)
        self.res.fit(X, y - self.lin.predict(X))
        return self

    def predict(self, X):
        return self.lin.predict(X) + self.res.predict(X)


def add_ew(feats, clean, cfg):
    f = feats.copy()
    for c in cfg.roll_features:
        x = clean[c]
        f[f"{c}_d3"] = x - x.shift(3)
        f[f"{c}_dev24"] = x - x.rolling(24, min_periods=24).mean()
        f[f"{c}_dmax24"] = x - x.rolling(24, min_periods=24).max()
        f[f"{c}_dmin24"] = x - x.rolling(24, min_periods=24).min()
    return f


VARIANTS = {
    "base":      dict(ew=False, make=lambda: make_model("ridge")),
    "ew":        dict(ew=True,  make=lambda: make_model("ridge")),
    "log":       dict(ew=False, make=lambda: LogTarget(make_model("ridge"))),
    "ew_log":    dict(ew=True,  make=lambda: LogTarget(make_model("ridge"))),
    "hybrid":    dict(ew=False, make=lambda: Hybrid()),
    "ew_hybrid": dict(ew=True,  make=lambda: Hybrid()),
}


def datasets(cleaner, raw, samples, si, cfg):
    clean = cleaner.transform(raw)
    f2 = build_sensor_features(clean, raw, cfg, False, True)
    f2ew = add_ew(f2, clean, cfg)
    f0 = build_sensor_features(clean, raw, cfg, False, False)
    return {"plain": assemble(f2, samples, si, cfg, True),
            "ew": assemble(f2ew, samples, si, cfg, True),
            "f0h": assemble(f0, samples, si, cfg, True)}


def main():
    cfg = Config(n_history=3)
    _, raw, si, samples = load_universe()
    n = len(samples)
    n_dev = int(n * (1 - TEST_FRACTION))
    dev = samples.iloc[:n_dev]
    bounds = [int(n_dev * b) for b in CV_BOUNDS]
    rows, pooled = [], {k: [] for k in list(VARIANTS) + ["xgb_ref"]}
    t0 = time.time()
    for k in range(len(bounds) - 1):
        tr_end, va_end = bounds[k], bounds[k + 1]
        cleaner = SensorCleaner().fit(raw.loc[:dev.dt.iloc[tr_end - 1]], cfg.sensors)
        D = datasets(cleaner, raw, dev, si, cfg)
        runs = [(name, v["make"], D["ew" if v["ew"] else "plain"]) for name, v in VARIANTS.items()]
        runs.append(("xgb_ref", lambda: make_model("xgb_l2"), D["f0h"]))
        for name, mk, d in runs:
            cols = split_cols(d, cfg)
            tr, va = d.iloc[:tr_end], d.iloc[tr_end:va_end]
            p = mk().fit(tr[cols], tr["Si"]).predict(va[cols])
            r = evaluate(va["Si"], p)
            r.update(fold=k + 1, variant=name)
            rows.append(r)
            pooled[name].append(pd.DataFrame({"y": va["Si"].values, "p": p}))
        print(f"fold {k+1} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/cv_peaks_folds.csv", index=False)
    s = df.groupby("variant")[["MAE", "R2", "within_0.10", "R2_normal", "MAE_normal"]].agg(["mean", "std"]).round(4)
    s.columns = ["_".join(c) for c in s.columns]
    # pooled error on upset samples (true Si > 1.2) across all validation folds
    up = {}
    for name, parts in pooled.items():
        a = pd.concat(parts)
        m = a.y > 1.2
        up[name] = (np.abs(a.y[m] - a.p[m]).mean(), int(m.sum()))
    s["MAE_upset(>1.2)"] = [round(up[i][0], 3) for i in s.index]
    s["n_upset"] = [up[i][1] for i in s.index]
    s = s.sort_values("MAE_mean")
    s.to_csv(f"{OUT_DIR}/cv_peaks_summary.csv")
    print("\nCV (new variants)\n", s.to_string())

    # ---- single pre-specified test evaluation ----
    best_new = s.drop(index=["xgb_ref", "base"], errors="ignore").MAE_mean.idxmin()
    print(f"\nSelected by CV MAE among new variants: {best_new}")
    cleaner = SensorCleaner().fit(raw.loc[:dev.dt.iloc[-1]], cfg.sensors)
    D = datasets(cleaner, raw, samples, si, cfg)
    out, preds = [], {}
    for name in ["base", best_new, "xgb_ref"]:
        v = VARIANTS.get(name)
        if name == "xgb_ref":
            d, mk = D["f0h"], (lambda: make_model("xgb_l2"))
        else:
            d, mk = D["ew" if v["ew"] else "plain"], v["make"]
        cols = split_cols(d, cfg)
        tr, te = d.iloc[:n_dev], d.iloc[n_dev:]
        p = mk().fit(tr[cols], tr["Si"]).predict(te[cols])
        preds[name] = p
        r = evaluate(te["Si"], p)
        m = te["Si"].values > 1.2
        r["MAE_upset(>1.2)"] = float(np.abs(te["Si"].values[m] - p[m]).mean())
        r["variant"] = name
        out.append(r)
    t = pd.DataFrame(out).set_index("variant")[["MAE", "RMSE", "R2", "within_0.10", "R2_normal", "MAE_normal",
                                                "MAE_upset(>1.2)"]].round(4)
    t.to_csv(f"{OUT_DIR}/test_peaks_results.csv")
    pr = te[["dt", "Si"]].copy()
    for k, v in preds.items():
        pr[k] = v
    pr["persistence"] = te["si_prev1"].values
    pr.to_csv(f"{OUT_DIR}/test_peaks_predictions.csv", index=False)
    print("\nTEST\n", t.to_string())


if __name__ == "__main__":
    main()
