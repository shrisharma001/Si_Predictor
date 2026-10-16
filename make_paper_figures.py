"""
make_paper_figures.py - redraws every figure of the paper at single-column size (3.4 in wide, 300 dpi)
and writes results/test_hybrid_predictions.csv.  Needs data/ and the results/ CSVs.

  python make_paper_figures.py      ->  paper/figs/fig_*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from run_experiments import DATA_FILE, OUT_DIR, TEST_FRACTION
from run_extra import load_universe
try:
    from si_pipeline import require_version
except ImportError:
    raise SystemExit("Your si_pipeline.py is out of date: copy the latest si_pipeline.py from the project folder over yours.")
require_version("1.3")
from si_pipeline import SiPipeline

FIG = "paper/figs/"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6, "axes.titlesize": 8})
W = 3.4

# ------------------------------------------------------------------ 1. pipeline diagram
def pipeline():
    steps = [("Raw data", "27 hourly variables +\nirregular Si lab results"),
             ("Cleaning", "fault codes, fences, forward-fill,\nwinsorising (training period only)"),
             ("Features", "sensors, rolling mean/std (3-24 h),\n3 previous Si results, time since last"),
             ("Models", "Ridge, XGBoost, Hybrid, GRU"),
             ("Evaluation", "time-ordered CV, final test period,\npaired block bootstrap"),
             ("Uncertainty", "90 % conformal intervals")]
    fig, ax = plt.subplots(figsize=(W, 2.45)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, len(steps) * 1.5)
    colors = ["#d9e6f2", "#e2f0d9", "#fff2cc", "#fce4d6", "#e4dfec", "#ededed"]
    for i, ((t, d), c) in enumerate(zip(steps, colors)):
        y = (len(steps) - 1 - i) * 1.5
        ax.add_patch(FancyBboxPatch((0.2, y + 0.1), 9.6, 1.15, boxstyle="round,pad=0.02,rounding_size=0.15", fc=c, ec="#555", lw=0.6))
        ax.text(0.5, y + 0.67, t, fontweight="bold", va="center", fontsize=7.8)
        ax.text(3.4, y + 0.67, d, va="center", fontsize=6.2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(5, y - 0.02), xytext=(5, y + 0.1), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#333"))
    fig.savefig(FIG + "fig_pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

# ------------------------------------------------------------------ 2. split effect
def split_effect():
    s = pd.read_csv(OUT_DIR + "/split_comparison.csv")
    order = ["random", "block 1 day", "block 7 days", "block 30 days", "chronological"]
    names = {"Sensors only - RF": ("Sensors only, RF", "tab:blue"), "Sensors only - XGBoost": ("Sensors only, XGBoost", "tab:green"),
             "Sensors + prev Si - Ridge": ("Sensors + previous Si, Ridge", "tab:red")}
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for k, (lab, c) in names.items():
        d = s[s.model == k].set_index("setting").loc[order]
        ax.errorbar(range(5), d.R2_mean, yerr=d.R2_std.fillna(0), marker="o", ms=3, lw=0.9, capsize=2, label=lab, color=c)
    ax.set_xticks(range(5)); ax.set_xticklabels(["random", "1-day\nblocks", "7-day\nblocks", "30-day\nblocks", "chrono-\nlogical"])
    ax.set_ylabel("R$^2$ on held-out part"); ax.axhline(0, color="#999", lw=0.5); ax.set_ylim(-0.45, 1.0); ax.legend(frameon=False, fontsize=6.6, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG + "fig_split.png", dpi=300); plt.close(fig)

# ------------------------------------------------------------------ hybrid refit (dev -> test) + importance
def hybrid_predictions():
    sensor = pd.read_excel(DATA_FILE, sheet_name="data"); si = pd.read_excel(DATA_FILE, sheet_name="Si")
    sensor["dt"] = pd.to_datetime(sensor["dt"]); si["dt"] = pd.to_datetime(si["dt"])
    _, raw, sid, samples = load_universe()
    n_dev = int(len(samples) * (1 - TEST_FRACTION)); dev, test = samples.iloc[:n_dev], samples.iloc[n_dev:]
    pipe = SiPipeline(use_lags=False, use_roll=True, use_history=True, model="hybrid", calibrate=False)
    pipe.fit(sensor[sensor.dt <= dev.dt.iloc[-1]], si[si.dt <= dev.dt.iloc[-1]])
    out = pipe.predict_frame(sensor, test.dt.values, si_history=si)
    y = test.set_index("dt").loc[pd.DatetimeIndex(out.time), "Si"].values
    prev = sid.set_index("dt").Si.shift(1).reindex(pd.DatetimeIndex(out.time)).values
    res = pd.DataFrame({"dt": out.time, "Si": y, "hybrid": out.prediction.values, "persistence": prev})
    res.to_csv(OUT_DIR + "/test_hybrid_predictions.csv", index=False)
    return pipe, res

def test_series(res):
    big = res.loc[res.Si.idxmax(), "dt"]
    w = (res.dt >= big - pd.Timedelta(days=8)) & (res.dt <= big + pd.Timedelta(days=6))
    fig, ax = plt.subplots(2, 1, figsize=(W, 3.3))
    ax[0].plot(res.dt, res.Si, color="k", lw=0.4, label="Measured"); ax[0].plot(res.dt, res.hybrid, color="tab:red", lw=0.4, label="Hybrid")
    ax[0].set_ylabel("Si (wt%)"); ax[0].legend(frameon=False, ncol=2, loc="upper left"); ax[0].set_title("(a) Whole test period")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y")); ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    q = res[w]
    ax[1].plot(q.dt, q.Si, "k.-", lw=0.5, ms=2, label="Measured"); ax[1].plot(q.dt, q.hybrid, color="tab:red", lw=0.7, label="Hybrid")
    ax[1].plot(q.dt, q.persistence, color="tab:blue", lw=0.6, ls="--", label="Persistence")
    ax[1].set_ylabel("Si (wt%)"); ax[1].legend(frameon=False, ncol=1, loc="upper left", fontsize=6.4); ax[1].set_title("(b) Largest upset")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    for a in ax: plt.setp(a.get_xticklabels(), rotation=0)
    fig.tight_layout(); fig.savefig(FIG + "fig_hybrid_test.png", dpi=300); plt.close(fig)

def intervals():
    p = pd.read_csv(OUT_DIR + "/intervals_test_predictions.csv", parse_dates=["dt"])
    big = p.loc[p.Si.idxmax(), "dt"]; start = big + pd.Timedelta(days=57)
    w1 = (p.dt >= start) & (p.dt < start + pd.Timedelta(days=4)); w2 = (p.dt >= big - pd.Timedelta(days=5)) & (p.dt <= big + pd.Timedelta(days=4))
    fig, ax = plt.subplots(2, 1, figsize=(W, 3.3))
    for a, w, t, step in [(ax[0], w1, "(a) Normal operation", 1), (ax[1], w2, "(b) Upset: intervals do not cover the peaks", 2)]:
        q = p[w]
        a.fill_between(q.dt, q.cqr_lo, q.cqr_hi, color="tab:orange", alpha=0.3, label="CQR")
        a.fill_between(q.dt, q.conformal_lo, q.conformal_hi, color="tab:blue", alpha=0.3, label="Conformal")
        a.plot(q.dt, q.Si, "k.-", lw=0.5, ms=2, label="Measured"); a.plot(q.dt, q.pred, color="tab:red", lw=0.7, label="Prediction")
        a.set_ylabel("Si (wt%)"); a.set_title(t); a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); a.xaxis.set_major_locator(mdates.DayLocator(interval=step))
    ax[1].legend(frameon=False, ncol=1, loc="upper right", fontsize=6.2)
    fig.tight_layout(); fig.savefig(FIG + "fig_intervals.png", dpi=300); plt.close(fig)

def learning():
    d = pd.read_csv(OUT_DIR + "/learning_curve_summary.csv")
    fig, ax = plt.subplots(figsize=(W, 2.1))
    for m, c in [("Ridge", "tab:red"), ("Hybrid", "tab:purple")]:
        q = d[d.model == m]
        ax.plot(q["size"], q.MAE, marker="o", ms=3, lw=0.9, color=c, label=m)
    ax.set_xlabel("Training samples (most recent)"); ax.set_ylabel("MAE on next block (wt%)"); ax.legend(frameon=False, fontsize=6.6)
    fig.tight_layout(); fig.savefig(FIG + "fig_learning.png", dpi=300); plt.close(fig)

def importance(pipe):
    names = pipe.feature_names_
    lin = pipe.model_.lin.named_steps["ridge"].coef_
    xgb = pipe.model_.res.feature_importances_
    def pretty(n):
        base = {"si_prev1": "Si, previous result", "si_prev2": "Si, 2nd previous", "si_prev3": "Si, 3rd previous",
                "si_prev_mean": "Si, mean of last 3", "hours_since_prev": "hours since last Si"}
        if n in base: return base[n]
        for suf, txt in [("_rm", " mean "), ("_rs", " std ")]:
            if suf in n: v, w = n.split(suf); return f"{v}{txt}{w} h"
        return n
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7))
    i = np.argsort(-np.abs(lin))[:8][::-1]; ax[0].barh([pretty(names[j]) for j in i], lin[i], color=["tab:red" if v > 0 else "tab:blue" for v in lin[i]])
    ax[0].set_title("(a) Ridge coefficients", fontsize=6.5); ax[0].set_xticks([-0.05, 0.05]); ax[0].axvline(0, color="#777", lw=0.5)
    k = np.argsort(-xgb)[:8][::-1]; ax[1].barh([pretty(names[j]) for j in k], xgb[k], color="tab:green"); ax[1].set_title("(b) XGBoost", fontsize=6.5)
    for a in ax: a.tick_params(axis="y", labelsize=6.6)
    fig.tight_layout(); fig.savefig(FIG + "fig_importance.png", dpi=300); plt.close(fig)
    pd.DataFrame({"feature": names, "ridge_coef": lin, "xgb_importance": xgb}).to_csv(OUT_DIR + "/hybrid_feature_importance.csv", index=False)

if __name__ == "__main__":
    pipeline(); split_effect(); intervals(); learning()
    pipe, res = hybrid_predictions(); test_series(res); importance(pipe)
    print("figures written to", FIG)