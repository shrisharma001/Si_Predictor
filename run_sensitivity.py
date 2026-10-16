"""run_sensitivity.py - one-at-a-time hyperparameter sensitivity of the hybrid model (CV only, test set untouched).
The deployed hyperparameters were fixed in advance, not tuned; this checks how much they matter.

  python run_sensitivity.py
"""
import time
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from run_experiments import CV_BOUNDS, OUT_DIR, TEST_FRACTION, split_cols
from run_extra import load_universe
from si_pipeline import Config, SensorCleaner, XGBEarlyStop, assemble, build_sensor_features, evaluate

DEFAULT = dict(ridge_alpha=10.0, max_depth=4, min_child_weight=5, learning_rate=0.03, colsample_bytree=0.6, reg_lambda=5.0, subsample=0.8)

class XGBP(XGBEarlyStop):
    def __init__(self, p):
        super().__init__("reg:squarederror"); self.p = p
    def _make(self, n_estimators, base_score, early=None):
        kw = dict(n_estimators=n_estimators, learning_rate=self.p["learning_rate"], max_depth=self.p["max_depth"],
                  min_child_weight=self.p["min_child_weight"], subsample=self.p["subsample"],
                  colsample_bytree=self.p["colsample_bytree"], reg_lambda=self.p["reg_lambda"],
                  objective="reg:squarederror", base_score=base_score, random_state=42, n_jobs=-1)
        if early: kw["early_stopping_rounds"] = early
        return XGBRegressor(**kw)

class HybridP:
    def __init__(self, p):
        self.lin = make_pipeline(StandardScaler(), Ridge(alpha=p["ridge_alpha"])); self.res = XGBP(p)
    def fit(self, X, y):
        self.lin.fit(X, y); self.res.fit(X, y - self.lin.predict(X)); return self
    def predict(self, X): return self.lin.predict(X) + self.res.predict(X)

variants = [("default", {})]
for k, vals in [("ridge_alpha", [1.0, 100.0, 1000.0]), ("max_depth", [3, 6]), ("min_child_weight", [1, 20]),
                ("learning_rate", [0.01, 0.1]), ("colsample_bytree", [0.3, 1.0]), ("reg_lambda", [1.0, 20.0]), ("subsample", [0.5, 1.0])]:
    for v in vals: variants.append((f"{k}={v}", {k: v}))

cfg = Config(n_history=3)
_, raw, si, samples = load_universe()
n_dev = int(len(samples) * (1 - TEST_FRACTION)); dev = samples.iloc[:n_dev]
bounds = [int(n_dev * b) for b in CV_BOUNDS]
rows, t0 = [], time.time()
for k in range(len(bounds) - 1):
    a, b = bounds[k], bounds[k + 1]
    cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[a - 1]], cfg.sensors)
    d = assemble(build_sensor_features(cl.transform(raw), raw, cfg, False, True), dev, si, cfg, True)
    cols = split_cols(d, cfg); tr, va = d.iloc[:a], d.iloc[a:b]
    for name, over in variants:
        p = {**DEFAULT, **over}
        r = evaluate(va.Si, HybridP(p).fit(tr[cols], tr.Si).predict(va[cols])); r.update(fold=k + 1, variant=name); rows.append(r)
    print(f"fold {k+1} done ({(time.time()-t0)/60:.1f} min)", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT_DIR}/sensitivity_cv_folds.csv", index=False)
s = df.groupby("variant")[["MAE", "R2", "R2_normal"]].mean().round(4)
pf = df.pivot(index="fold", columns="variant", values="MAE")
s["folds_better_than_default"] = [(pf[v] < pf["default"]).sum() for v in s.index]
s["MAE_change_vs_default_%"] = ((s.MAE / s.loc["default", "MAE"] - 1) * 100).round(2)
s = s.loc[[v for v, _ in variants]]; s.to_csv(f"{OUT_DIR}/sensitivity_cv_summary.csv")
print(s.to_string())
