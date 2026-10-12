"""
run_split_comparison.py - how much does the evaluation protocol change the reported accuracy?

Everything happens INSIDE THE DEVELOPMENT PERIOD (first 85 % of samples); the final test set is not used.
For the same models and features we compare how samples are assigned to a 80 % train / 20 % test split:

  random        : every sample assigned independently (the common mistake on time series)
  block 1 day   : whole 1-day blocks are assigned together
  block 7 days  : whole 7-day blocks
  block 30 days : whole 30-day blocks
  chronological : the last 20 % in time is the test set (the protocol used in the paper)

The sensor cleaner is fitted once on the whole development sensor stream for ALL settings,
so the only thing that differs between settings is how the split is made.
Random / block settings are repeated with 5 seeds (mean +- std); chronological is deterministic.

  python run_split_comparison.py
"""
import os
import time

import numpy as np
import pandas as pd

from run_experiments import OUT_DIR, TEST_FRACTION, split_cols
from run_extra import load_universe
from si_pipeline import Config, SensorCleaner, assemble, build_sensor_features, evaluate, make_model

TEST_SHARE = 0.20
SEEDS = range(5)
SETTINGS = [("random", 0), ("block 1 day", 1), ("block 7 days", 7), ("block 30 days", 30), ("chronological", None)]
MODELS = [("Sensors only - RF", "F0", "rf"),
          ("Sensors only - XGBoost", "F0", "xgb_l2"),
          ("Sensors + prev Si - Ridge", "F2+H", "ridge")]


def pick_test_mask(dt, block_days, seed):
    n = len(dt)
    rng = np.random.default_rng(seed)
    if block_days is None:                                   # chronological
        m = np.zeros(n, bool)
        m[int(n * (1 - TEST_SHARE)):] = True
        return m
    if block_days == 0:                                      # fully random samples
        m = np.zeros(n, bool)
        m[rng.choice(n, int(n * TEST_SHARE), replace=False)] = True
        return m
    bid = ((dt - dt.min()) / pd.Timedelta(days=block_days)).astype(int).values
    ids = np.unique(bid)
    rng.shuffle(ids)
    m = np.zeros(n, bool)
    for b in ids:                                            # add whole blocks until 20 % reached
        m |= bid == b
        if m.mean() >= TEST_SHARE:
            break
    return m


def main():
    cfg = Config(n_history=3)
    _, raw, si, samples = load_universe()
    n = len(samples)
    n_dev = int(n * (1 - TEST_FRACTION))
    dev = samples.iloc[:n_dev]
    cleaner = SensorCleaner().fit(raw.loc[:dev.dt.iloc[-1]], cfg.sensors)
    clean = cleaner.transform(raw)
    data = {"F0": assemble(build_sensor_features(clean, raw, cfg, False, False), dev, si, cfg, False),
            "F2+H": assemble(build_sensor_features(clean, raw, cfg, False, True), dev, si, cfg, True)}
    print(f"[INFO] development samples: {n_dev}  ({dev.dt.min().date()} -> {dev.dt.max().date()})")
    rows, t0 = [], time.time()
    for sname, bd in SETTINGS:
        for mlabel, fs, mname in MODELS:
            d = data[fs]
            cols = split_cols(d, cfg)
            seeds = [0] if bd is None else list(SEEDS)
            for seed in seeds:
                te_mask = pick_test_mask(d["dt"], bd, seed)
                tr, te = d[~te_mask], d[te_mask]
                p = make_model(mname).fit(tr[cols], tr["Si"]).predict(te[cols])
                r = evaluate(te["Si"], p)
                r.update(setting=sname, model=mlabel, seed=seed, test_rows=int(te_mask.sum()))
                rows.append(r)
            print(f"{sname:14s} {mlabel:27s} done ({(time.time()-t0)/60:.1f} min)", flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT_DIR}/split_comparison_runs.csv", index=False)

    df = pd.DataFrame(rows)
    s = df.groupby(["model", "setting"], sort=False)[["R2", "MAE", "R2_normal"]].agg(["mean", "std"]).round(4)
    s.columns = ["_".join(c) for c in s.columns]
    s.to_csv(f"{OUT_DIR}/split_comparison.csv")
    print("\n", s.to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    order = [x[0] for x in SETTINGS]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for k, metric in enumerate(["R2", "MAE"]):
        for (mlabel, _, _), c in zip(MODELS, ["tab:blue", "tab:green", "tab:red"]):
            sub = s.loc[mlabel].loc[order]
            ax[k].errorbar(range(len(order)), sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"].fillna(0),
                           marker="o", capsize=2, label=mlabel, color=c)
        ax[k].set_xticks(range(len(order)))
        ax[k].set_xticklabels([o.replace(" ", "\n", 1) for o in order], fontsize=6.5)
        ax[k].set_ylabel(metric if metric == "MAE" else "R$^2$")
    ax[0].set_title("(a) R$^2$ by split method")
    ax[1].set_title("(b) MAE by split method")
    ax[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_split_leakage.png", dpi=300)
    print("saved fig_split_leakage.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    main()
