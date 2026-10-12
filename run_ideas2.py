"""run_ideas2.py - CV-only: sliding-window training, error feedback from the previous cast, simple blend."""
import time
import numpy as np, pandas as pd
from run_experiments import CV_BOUNDS, OUT_DIR, TEST_FRACTION, split_cols
from run_extra import load_universe
from run_peaks import Hybrid
from si_pipeline import Config, SensorCleaner, assemble, build_sensor_features, evaluate, make_model

def feedback_pred(model, tr, va, cols, lags=2):
    ptr, pva = model.predict(tr[cols]), model.predict(va[cols])
    rtr, rva = tr.Si.values - ptr, va.Si.values - pva
    r = np.concatenate([rtr, rva]); n = len(rtr)
    R = np.column_stack([np.r_[[np.nan] * k, r[:-k]] for k in range(1, lags + 1)])
    phi = np.linalg.lstsq(R[lags:n], rtr[lags:], rcond=None)[0]
    return pva + R[n:] @ phi, phi

cfg = Config(n_history=3)
_, raw, si, samples = load_universe()
n_dev = int(len(samples) * (1 - TEST_FRACTION)); dev = samples.iloc[:n_dev]
bounds = [int(n_dev * b) for b in CV_BOUNDS]
VARS = ["ref_hybrid", "window_3000", "window_6000", "ridge", "ridge+feedback(2 lags)", "blend(ridge,xgb,rf)"]
rows = []; phis = []; t0 = time.time()
for k in range(len(bounds) - 1):
    a, b = bounds[k], bounds[k + 1]
    cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[a - 1]], cfg.sensors)
    d = assemble(build_sensor_features(cl.transform(raw), raw, cfg, False, True), dev, si, cfg, True)
    cols = split_cols(d, cfg); tr, va = d.iloc[:a], d.iloc[a:b]; y = va.Si.values
    P = {}
    P["ref_hybrid"] = Hybrid().fit(tr[cols], tr.Si).predict(va[cols])
    for W in (3000, 6000):
        t = tr.iloc[-W:]; P[f"window_{W}"] = Hybrid().fit(t[cols], t.Si).predict(va[cols])
    rid = make_model("ridge").fit(tr[cols], tr.Si); P["ridge"] = rid.predict(va[cols])
    P["ridge+feedback(2 lags)"], phi = feedback_pred(rid, tr, va, cols); phis.append(phi)
    xg = make_model("xgb_l2").fit(tr[cols], tr.Si); rf = make_model("rf").fit(tr[cols], tr.Si)
    P["blend(ridge,xgb,rf)"] = (P["ridge"] + xg.predict(va[cols]) + rf.predict(va[cols])) / 3
    for name in VARS:
        r = evaluate(y, P[name]); r.update(fold=k + 1, variant=name); rows.append(r)
    print(f"fold {k+1} done ({(time.time()-t0)/60:.1f} min)", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT_DIR}/ideas2_cv_folds.csv", index=False)
s = df.groupby("variant")[["MAE", "R2", "within_0.10", "R2_normal"]].agg(["mean", "std"]).round(4).loc[VARS]
s.columns = ["_".join(c) for c in s.columns]; s.to_csv(f"{OUT_DIR}/ideas2_cv_summary.csv")
print(s[["MAE_mean", "MAE_std", "R2_mean", "R2_normal_mean", "within_0.10_mean"]].to_string())
pf = df.pivot(index="fold", columns="variant", values="MAE")
for v in VARS:
    if v != "ref_hybrid": print(f"{v:26s} lower MAE than ref_hybrid in {(pf[v] < pf['ref_hybrid']).sum()} of 5 folds")
print("feedback coefficients (lag1, lag2) per fold:", [list(np.round(p, 3)) for p in phis])
