"""
run_extra.py  -  second round of experiments (same protocol as run_experiments.py)

  python run_extra.py --stage history     # CV: how many previous Si results to use + MLP baseline
  python run_extra.py --stage classify    # CV (+ one test evaluation): low / normal / high bands
  python run_extra.py --stage analysis    # paired block-bootstrap tests, error analysis, figures
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from run_experiments import CV_BOUNDS, DATA_FILE, OUT_DIR, TEST_FRACTION, split_cols
from si_pipeline import (Config, SensorCleaner, assemble, build_sensor_features, evaluate,
                         make_model, prepare_sensor_frame, prepare_si_frame, valid_samples)

HIST_SWEEP = (1, 3, 5, 10)


def load_universe(max_hist=10):
    cfg = Config(n_history=max_hist)
    raw = prepare_sensor_frame(pd.read_excel(DATA_FILE, sheet_name="data"), cfg)
    si = prepare_si_frame(pd.read_excel(DATA_FILE, sheet_name="Si"), cfg)
    samples = valid_samples(si, raw.index, cfg)          # same rows for every n
    return cfg, raw, si, samples


def mlp():
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                                      validation_fraction=0.15, max_iter=300, random_state=42))


# ----------------------------------------------------------------------------
def stage_history():
    _, raw, si, samples = load_universe()
    n = len(samples)
    n_dev = int(n * (1 - TEST_FRACTION))
    dev = samples.iloc[:n_dev]
    bounds = [int(n_dev * b) for b in CV_BOUNDS]
    rows = []
    t0 = time.time()
    for k in range(len(bounds) - 1):
        tr_end, va_end = bounds[k], bounds[k + 1]
        cleaner = SensorCleaner().fit(raw.loc[:dev.dt.iloc[tr_end - 1]], Config().sensors)
        feats = build_sensor_features(cleaner.transform(raw), raw, Config(), False, True)   # F2
        for nh in HIST_SWEEP:
            cfgn = Config(n_history=nh)
            d = assemble(feats, dev, si, cfgn, True)
            cols = split_cols(d, cfgn)
            tr, va = d.iloc[:tr_end], d.iloc[tr_end:va_end]
            models = {"ridge": make_model("ridge"), "xgb_l2": make_model("xgb_l2")}
            if nh == 3:
                models["mlp"] = mlp()
            for name, m in models.items():
                m.fit(tr[cols], tr["Si"])
                r = evaluate(va["Si"], m.predict(va[cols]))
                r.update(fold=k + 1, n_history=nh, model=name, n_features=len(cols))
                rows.append(r)
        pd.DataFrame(rows).to_csv(f"{OUT_DIR}/cv_history_folds.csv", index=False)
        print(f"fold {k+1} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    s = (df.groupby(["model", "n_history"])[["MAE", "R2", "within_0.10", "R2_normal", "MAE_normal"]]
           .agg(["mean", "std"]).round(4))
    s.columns = ["_".join(c) for c in s.columns]
    s.to_csv(f"{OUT_DIR}/cv_history_summary.csv")
    print(s.to_string())


# ----------------------------------------------------------------------------
def band_labels(y, lo, hi):
    return np.where(y < lo, 0, np.where(y > hi, 2, 1))


def clf_metrics(yt, yp):
    return {"macro_F1": f1_score(yt, yp, average="macro"),
            "bal_acc": balanced_accuracy_score(yt, yp),
            "acc": accuracy_score(yt, yp),
            "recall_low": float(((yp == 0) & (yt == 0)).sum() / max((yt == 0).sum(), 1)),
            "recall_high": float(((yp == 2) & (yt == 2)).sum() / max((yt == 2).sum(), 1))}


def run_band_models(tr, te, cols, lo, hi):
    ytr_b, yte_b = band_labels(tr["Si"].values, lo, hi), band_labels(te["Si"].values, lo, hi)
    out = {}
    out["majority"] = np.full(len(te), np.bincount(ytr_b).argmax())
    out["persistence"] = band_labels(te["si_prev1"].values, lo, hi)
    for name in ("ridge", "xgb_l2"):
        m = make_model(name).fit(tr[cols], tr["Si"])
        out[f"{name} (regress->band)"] = band_labels(m.predict(te[cols]), lo, hi)
    # class weights so the rarer low/high bands are not ignored
    w = pd.Series(ytr_b).map(len(ytr_b) / (3 * pd.Series(ytr_b).value_counts())).values
    c = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.6, min_child_weight=5, reg_lambda=5.0,
                      objective="multi:softprob", random_state=42, n_jobs=-1)
    c.fit(tr[cols], ytr_b, sample_weight=w)
    out["xgb_classifier"] = c.predict(te[cols])
    return yte_b, out


def stage_classify():
    cfgn = Config(n_history=3)
    _, raw, si, samples = load_universe()
    n = len(samples)
    n_dev = int(n * (1 - TEST_FRACTION))
    dev = samples.iloc[:n_dev]
    bounds = [int(n_dev * b) for b in CV_BOUNDS]
    rows = []
    for k in range(len(bounds) - 1):
        tr_end, va_end = bounds[k], bounds[k + 1]
        cleaner = SensorCleaner().fit(raw.loc[:dev.dt.iloc[tr_end - 1]], cfgn.sensors)
        feats = build_sensor_features(cleaner.transform(raw), raw, cfgn, False, True)
        d = assemble(feats, dev, si, cfgn, True)
        cols = split_cols(d, cfgn)
        tr, va = d.iloc[:tr_end], d.iloc[tr_end:va_end]
        lo, hi = tr["Si"].quantile([0.25, 0.75])          # thresholds from TRAIN only
        yb, preds = run_band_models(tr, va, cols, lo, hi)
        for name, p in preds.items():
            r = clf_metrics(yb, p)
            r.update(fold=k + 1, model=name)
            rows.append(r)
        print(f"fold {k+1} done", flush=True)
    df = pd.DataFrame(rows)
    s = df.groupby("model")[["macro_F1", "bal_acc", "acc", "recall_low", "recall_high"]].agg(["mean", "std"]).round(3)
    s.columns = ["_".join(c) for c in s.columns]
    s.to_csv(f"{OUT_DIR}/cv_band_summary.csv")
    print("\nCV band classification\n", s.to_string())

    # one-off test evaluation (thresholds = dev quartiles)
    cleaner = SensorCleaner().fit(raw.loc[:dev.dt.iloc[-1]], cfgn.sensors)
    feats = build_sensor_features(cleaner.transform(raw), raw, cfgn, False, True)
    d = assemble(feats, samples, si, cfgn, True)
    cols = split_cols(d, cfgn)
    tr, te = d.iloc[:n_dev], d.iloc[n_dev:]
    lo, hi = tr["Si"].quantile([0.25, 0.75])
    yb, preds = run_band_models(tr, te, cols, lo, hi)
    rows = []
    for name, p in preds.items():
        r = clf_metrics(yb, p)
        r["model"] = name
        rows.append(r)
    t = pd.DataFrame(rows).set_index("model").round(3)
    t.to_csv(f"{OUT_DIR}/test_band_results.csv")
    print(f"\nTEST band classification (low < {lo:.3f}, high > {hi:.3f}; "
          f"class shares {np.bincount(yb)/len(yb)})\n", t.to_string())


# ----------------------------------------------------------------------------
def block_bootstrap_diff(e_a, e_b, block=50, n_boot=3000, seed=0):
    """CI for mean(e_a) - mean(e_b), resampling contiguous blocks (keeps autocorrelation)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(e_a) - np.asarray(e_b)
    n = len(d)
    nb = int(np.ceil(n / block))
    starts_max = n - block
    stats = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, starts_max + 1, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        stats[i] = d[idx].mean()
    return d.mean(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def stage_analysis():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = pd.read_csv(f"{OUT_DIR}/final_test_predictions.csv", parse_dates=["dt"])
    y = p["Si"].values
    P = {c: p[c].values for c in p.columns if ":" in c}
    ridge, xgbh, pers = P["F2+H:ridge"], P["F0+H:xgb_l2"], P["F0+H:persistence"]
    sens, mean = P["F1:xgb_l2"], P["F0:mean"]

    # --- paired block-bootstrap tests on absolute error ------------------------------------
    pairs = [("ridge(F2+H) vs persistence", ridge, pers),
             ("xgb(F0+H) vs persistence", xgbh, pers),
             ("ridge(F2+H) vs xgb(F0+H)", ridge, xgbh),
             ("sensors-only xgb(F1) vs mean", sens, mean),
             ("sensors+history ridge vs sensors-only xgb", ridge, sens)]
    rows = []
    for name, a, b in pairs:
        d, lo, hi = block_bootstrap_diff(np.abs(y - a), np.abs(y - b))
        rows.append({"comparison (A vs B)": name, "MAE_A-MAE_B": d, "CI95_low": lo, "CI95_high": hi,
                     "significant": not (lo <= 0 <= hi)})
    bt = pd.DataFrame(rows).round(4)
    bt.to_csv(f"{OUT_DIR}/bootstrap_tests.csv", index=False)
    print("Paired block bootstrap (negative = A better)\n", bt.to_string(index=False))

    # --- error by Si regime ---------------------------------------------------------------
    bins = [-np.inf, 0.3, 0.6, 1.2, np.inf]
    labels = ["<0.3", "0.3-0.6", "0.6-1.2", ">1.2"]
    reg = pd.cut(y, bins, labels=labels)
    t = pd.DataFrame({"regime": reg, "n": 1, "persistence": np.abs(y - pers), "ridge": np.abs(y - ridge),
                      "xgb": np.abs(y - xgbh), "mean": np.abs(y - mean)})
    g = t.groupby("regime", observed=True).agg(n=("n", "sum"), persistence=("persistence", "mean"),
                                               ridge=("ridge", "mean"), xgb=("xgb", "mean"),
                                               mean=("mean", "mean")).round(4)
    g["share_%"] = (g.n / g.n.sum() * 100).round(1)
    g.to_csv(f"{OUT_DIR}/error_by_regime.csv")
    print("\nTest MAE by true-Si regime\n", g.to_string())

    # R2 with / without upset samples
    keep = (y <= 1.2) & (y >= 0.1)
    ex = {k: evaluate(y[keep], v[keep]) for k, v in [("persistence", pers), ("ridge", ridge), ("xgb", xgbh)]}
    print("\nExcluding upsets (true Si in [0.1,1.2]): ",
          {k: (round(v["R2"], 3), round(v["MAE"], 4)) for k, v in ex.items()}, f"n={keep.sum()}")

    # --- figures ---------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    # (1) test period time series, zoomed on the upset
    big = p.loc[p["Si"].idxmax(), "dt"]
    w = (p.dt >= big - pd.Timedelta(days=21)) & (p.dt <= big + pd.Timedelta(days=21))
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.2))
    ax[0].plot(p.dt, y, lw=0.6, color="k", label="Measured Si")
    ax[0].plot(p.dt, ridge, lw=0.6, color="tab:red", alpha=0.8, label="Ridge (sensors + previous Si)")
    ax[0].set_ylabel("Si (wt%)"); ax[0].legend(loc="upper left", frameon=False); ax[0].set_title("(a) Full test period")
    ax[1].plot(p.dt[w], y[w], lw=0.8, color="k", marker=".", ms=2, label="Measured")
    ax[1].plot(p.dt[w], ridge[w], lw=0.8, color="tab:red", label="Ridge")
    ax[1].plot(p.dt[w], pers[w], lw=0.8, color="tab:blue", ls="--", label="Persistence")
    ax[1].set_ylabel("Si (wt%)"); ax[1].legend(frameon=False); ax[1].set_title("(b) Zoom on the largest upset")
    import matplotlib.dates as mdates
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y")); ax[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    for a in ax:
        plt.setp(a.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_test_timeseries.png", dpi=300); plt.close(fig)

    # (2) scatter predicted vs measured, normal range
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, sharey=True)
    for a, (lab, pr) in zip(ax, [("Sensors only (XGBoost)", sens), ("Sensors + previous Si (Ridge)", ridge)]):
        a.scatter(y, pr, s=3, alpha=0.3, color="tab:blue"); a.plot([0, 1.5], [0, 1.5], "k--", lw=0.8)
        a.set_xlim(0, 1.5); a.set_ylim(0, 1.5); a.set_xlabel("Measured Si (wt%)"); a.set_title(lab)
    ax[0].set_ylabel("Predicted Si (wt%)"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_scatter.png", dpi=300); plt.close(fig)

    # (3) CV summary bar chart
    s = pd.read_csv(f"{OUT_DIR}/cv_summary.csv")
    pick = [("F0", "mean"), ("F1", "xgb_l2"), ("F0+H", "persistence"), ("F0+H", "xgb_l2"), ("F2+H", "ridge")]
    lab = ["Mean", "Sensors only\n(XGBoost)", "Previous Si\n(persistence)", "Sensors + prev. Si\n(XGBoost)",
           "Sensors + prev. Si\n(Ridge)"]
    sub = [s[(s.features == f) & (s.model == m)].iloc[0] for f, m in pick]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(range(len(sub)), [r.MAE_mean for r in sub], yerr=[r.MAE_std for r in sub], capsize=3,
           color=["gray", "tab:blue", "tab:orange", "tab:green", "tab:red"])
    ax.set_xticks(range(len(sub))); ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel("CV MAE (wt% Si)"); ax.set_title("Time-series CV, mean ± std over 5 folds")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig_cv_mae.png", dpi=300); plt.close(fig)
    print("\nfigures saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["history", "classify", "analysis"], required=True)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    {"history": stage_history, "classify": stage_classify, "analysis": stage_analysis}[a.stage]()
