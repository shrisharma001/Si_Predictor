"""run_learning_curve.py - how does MAE change with the amount of (most recent) training data? CV folds 3-5 only,
so every training size can be tested on a block that follows it. The test set is not used."""
import numpy as np, pandas as pd
from run_experiments import CV_BOUNDS, OUT_DIR, TEST_FRACTION, split_cols
from run_extra import load_universe
from run_peaks import Hybrid
from si_pipeline import Config, SensorCleaner, assemble, build_sensor_features, evaluate, make_model

SIZES = [500, 1000, 2000, 4000, 6000, 8000, 10000]
cfg = Config(n_history=3)
_, raw, si, samples = load_universe()
n_dev = int(len(samples) * (1 - TEST_FRACTION)); dev = samples.iloc[:n_dev]
bounds = [int(n_dev * b) for b in CV_BOUNDS]
rows = []
for k in (2, 3, 4):
    a, b = bounds[k], bounds[k + 1]
    cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[a - 1]], cfg.sensors)
    d = assemble(build_sensor_features(cl.transform(raw), raw, cfg, False, True), dev, si, cfg, True)
    cols = split_cols(d, cfg); va = d.iloc[a:b]
    for size in SIZES + [a]:
        if size > a: continue
        tr = d.iloc[a - size:a]
        for name, mk in [("Ridge", lambda: make_model("ridge")), ("Hybrid", Hybrid)]:
            r = evaluate(va.Si, mk().fit(tr[cols], tr.Si).predict(va[cols]))
            r.update(fold=k + 1, size=size, is_all=(size == a), model=name); rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f"{OUT_DIR}/learning_curve_folds.csv", index=False)
g = df[~df.is_all].groupby(["model", "size"]).agg(MAE=("MAE", "mean"), R2=("R2", "mean"), n_folds=("fold", "nunique")).round(4)
print(g.to_string())
allp = df[df.is_all].groupby(["model", "fold"]).agg(size=("size", "first"), MAE=("MAE", "mean")).round(4); print("\nall available data per fold:\n", allp.to_string())
g.to_csv(f"{OUT_DIR}/learning_curve_summary.csv")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for m, c in [("Ridge", "tab:red"), ("Hybrid", "tab:purple")]:
    s = g.loc[m].reset_index()
    ax.plot(s["size"], s.MAE, marker="o", color=c, label=m)
ax.set_xlabel("Training samples (most recent)"); ax.set_ylabel("MAE on following block"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_learning_curve.png", dpi=300)
