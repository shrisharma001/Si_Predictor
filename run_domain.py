"""run_domain.py - CV-only test of process-knowledge features (ratios and temperature-profile summaries)."""
import time
import numpy as np, pandas as pd
from run_experiments import CV_BOUNDS, OUT_DIR, TEST_FRACTION, split_cols
from run_extra import load_universe
from run_peaks import Hybrid
from si_pipeline import Config, SensorCleaner, assemble, build_sensor_features, evaluate, make_model

def add_domain(feats, c, rolling):
    f = feats.copy()
    fb = (c.Fb.clip(lower=1000) / 1000.0)                       # blast flow in 1000 m3/min (bounded: shutdowns)
    f["o2_frac"] = c.Fo / (60.0 * c.Fb.clip(lower=1000))        # oxygen share of the blast volume
    f["perm_idx"] = c.dP / fb ** 2                               # gas permeability proxy (pressure drop / flow^2)
    f["perm_idx_u"] = c.dPu / fb ** 2
    f["perm_idx_l"] = c.dPl / fb ** 2
    f["dPu_share"] = c.dPu / (c.dP.abs() + 0.05)
    f["blast_dp"] = (c.Ph - c.Pt) / fb ** 2
    tp = c[[f"Tp{i}" for i in range(1, 11)]]; tt = c[[f"Tt{i}" for i in range(1, 5)]]
    f["Tp_mean"], f["Tp_std"], f["Tp_max"], f["Tp_min"] = tp.mean(1), tp.std(1), tp.max(1), tp.min(1)
    f["Tt_mean"], f["Tt_spread"] = tt.mean(1), tt.max(1) - tt.min(1)
    f["Tp_minus_Tt"] = f["Tp_mean"] - f["Tt_mean"]
    if rolling:
        for b in ["o2_frac", "perm_idx", "Tp_mean", "Tp_std", "Tt_mean", "Tp_minus_Tt"]:
            for w in (6, 24):
                f[f"{b}_rm{w}"] = f[b].rolling(w, min_periods=w).mean()
    return f

cfg = Config(n_history=3)
_, raw, si, samples = load_universe()
n_dev = int(len(samples) * (1 - TEST_FRACTION)); dev = samples.iloc[:n_dev]
bounds = [int(n_dev * b) for b in CV_BOUNDS]
VARS = ["ref_hybrid", "ref_ridge", "hybrid+domain", "hybrid+domain+roll", "ridge+domain+roll"]
rows = []; t0 = time.time()
for k in range(len(bounds) - 1):
    a, b = bounds[k], bounds[k + 1]
    cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[a - 1]], cfg.sensors); clean = cl.transform(raw)
    f2 = build_sensor_features(clean, raw, cfg, False, True)
    D0 = assemble(f2, dev, si, cfg, True)
    D1 = assemble(add_domain(f2, clean, False), dev, si, cfg, True)
    D2 = assemble(add_domain(f2, clean, True), dev, si, cfg, True)
    for d in (D0, D1, D2): assert len(d) == n_dev, len(d)
    plan = {"ref_hybrid": (D0, Hybrid), "ref_ridge": (D0, lambda: make_model("ridge")),
            "hybrid+domain": (D1, Hybrid), "hybrid+domain+roll": (D2, Hybrid),
            "ridge+domain+roll": (D2, lambda: make_model("ridge"))}
    for name, (d, mk) in plan.items():
        cols = split_cols(d, cfg); tr, va = d.iloc[:a], d.iloc[a:b]
        r = evaluate(va.Si, mk().fit(tr[cols], tr.Si).predict(va[cols])); r.update(fold=k + 1, variant=name, nfeat=len(cols)); rows.append(r)
    print(f"fold {k+1} done ({(time.time()-t0)/60:.1f} min)", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT_DIR}/domain_cv_folds.csv", index=False)
s = df.groupby("variant")[["MAE", "R2", "within_0.10", "R2_normal"]].agg(["mean", "std"]).round(4).loc[VARS]
s.columns = ["_".join(c) for c in s.columns]; s.to_csv(f"{OUT_DIR}/domain_cv_summary.csv")
print(s[["MAE_mean", "MAE_std", "R2_mean", "R2_normal_mean", "within_0.10_mean"]].to_string())
pf = df.pivot(index="fold", columns="variant", values="MAE")
for v in VARS:
    if v not in ("ref_hybrid", "ref_ridge"):
        ref = "ref_ridge" if v.startswith("ridge") else "ref_hybrid"
        print(f"{v:22s} lower MAE than {ref} in {(pf[v] < pf[ref]).sum()} of 5 folds; mean MAE change {pf[v].mean()-pf[ref].mean():+.4f}")
print("n features:", df.groupby("variant").nfeat.first().to_dict())
