"""run_thermal.py - CV-only: do hot-blast-temperature (Th) based thermal-state features help?
Th = hot-blast temperature (deg C). Together with oxygen enrichment it drives flame temperature / furnace heat.
Moisture and coal injection are not in the data, so only a crude proxy is possible.
"""
import time
import numpy as np, pandas as pd
from run_experiments import CV_BOUNDS, OUT_DIR, TEST_FRACTION, split_cols
from run_extra import load_universe
from run_peaks import Hybrid
from si_pipeline import Config, SensorCleaner, assemble, build_sensor_features, evaluate, make_model

def add_thermal(feats, c):
    f = feats.copy()
    o2 = c.Fo / (60.0 * c.Fb.clip(lower=1000))
    f["raft_proxy"] = c.Th + 45.0 * (100.0 * o2)             # crude: ~45 C per 1 % oxygen enrichment
    f["Th_x_o2"] = c.Th * o2
    for w in (3, 6, 12):
        f[f"Th_d{w}"] = c.Th - c.Th.shift(w)
    f["Th_std24"] = c.Th.rolling(24, min_periods=24).std()
    f["raft_rm6"] = f["raft_proxy"].rolling(6, min_periods=6).mean()
    f["raft_rm24"] = f["raft_proxy"].rolling(24, min_periods=24).mean()
    f["raft_d6"] = f["raft_proxy"] - f["raft_proxy"].shift(6)
    return f

cfg = Config(n_history=3)
_, raw, si, samples = load_universe()
n_dev = int(len(samples) * (1 - TEST_FRACTION)); dev = samples.iloc[:n_dev]
bounds = [int(n_dev * b) for b in CV_BOUNDS]
rows = []
for k in range(len(bounds) - 1):
    a, b = bounds[k], bounds[k + 1]
    cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[a - 1]], cfg.sensors); clean = cl.transform(raw)
    f2 = build_sensor_features(clean, raw, cfg, False, True)
    D0 = assemble(f2, dev, si, cfg, True); D1 = assemble(add_thermal(f2, clean), dev, si, cfg, True)
    assert len(D0) == len(D1) == n_dev
    for name, d, mk in [("ref_hybrid", D0, Hybrid), ("hybrid+thermal(Th)", D1, Hybrid),
                        ("ref_ridge", D0, lambda: make_model("ridge")), ("ridge+thermal(Th)", D1, lambda: make_model("ridge"))]:
        cols = split_cols(d, cfg); tr, va = d.iloc[:a], d.iloc[a:b]
        r = evaluate(va.Si, mk().fit(tr[cols], tr.Si).predict(va[cols])); r.update(fold=k + 1, variant=name); rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f"{OUT_DIR}/thermal_cv_folds.csv", index=False)
s = df.groupby("variant")[["MAE", "R2", "within_0.10", "R2_normal"]].agg(["mean", "std"]).round(4)
s.columns = ["_".join(c) for c in s.columns]; s.to_csv(f"{OUT_DIR}/thermal_cv_summary.csv"); print(s.to_string())
pf = df.pivot(index="fold", columns="variant", values="MAE")
for v, ref in [("hybrid+thermal(Th)", "ref_hybrid"), ("ridge+thermal(Th)", "ref_ridge")]:
    print(f"{v}: lower MAE than {ref} in {(pf[v] < pf[ref]).sum()} of 5 folds; mean change {pf[v].mean()-pf[ref].mean():+.4f}")
