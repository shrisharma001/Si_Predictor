"""
run_ideas.py - diagnostics + CV-only test of new ideas. The held-out test set is NOT used.
"""
import time

import numpy as np
import pandas as pd

from run_experiments import CV_BOUNDS, DATA_FILE, OUT_DIR, TEST_FRACTION, split_cols
from run_extra import load_universe
from run_peaks import Hybrid
from si_pipeline import (Config, SensorCleaner, assemble, build_sensor_features, evaluate, make_model,
                         prepare_si_frame)


class DeltaTarget:
    """Predict the CHANGE from the previous Si result, then add it back."""
    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        self.base.fit(X, y - X["si_prev1"])
        return self

    def predict(self, X):
        return X["si_prev1"].values + self.base.predict(X)


def prev_state_deltas(d, feats, si_all, cfg):
    """Sensor state now minus sensor state at the time of the previous Si result."""
    st = pd.DatetimeIndex(si_all.dt).values
    idx = np.searchsorted(st, pd.DatetimeIndex(d.dt).values, side="left") - 1
    left = pd.DataFrame({"dt_prev": st[idx]})
    right = feats[cfg.sensors].reset_index().rename(columns={"dt": "dt_prev"})
    pv = pd.merge_asof(left, right, on="dt_prev", direction="backward", tolerance=pd.Timedelta(hours=1))
    out = d.copy()
    for c in cfg.sensors:
        out[f"{c}_dprev"] = (d[c].values - pv[c].values)
    return out.fillna(0.0)


def slow_features(feats, clean, cfg):
    f = feats.copy()
    for c in cfg.roll_features:
        for w in (48, 72):
            f[f"{c}_rm{w}"] = clean[c].rolling(w, min_periods=1).mean()
    return f


def main():
    cfg = Config(n_history=3)
    _, raw, si, samples = load_universe()
    n = len(samples)
    n_dev = int(n * (1 - TEST_FRACTION))
    dev = samples.iloc[:n_dev]
    bounds = [int(n_dev * b) for b in CV_BOUNDS]

    # ---------------- diagnostics on the Si series itself ----------------
    gaps = si.dt.diff().dt.total_seconds().div(3600).dropna()
    print("Si sampling gap (hours) quantiles:", gaps.quantile([.1, .25, .5, .75, .9, .99]).round(2).to_dict())
    v = si.Si.values
    g = si.dt.diff().dt.total_seconds().div(3600).values
    dv = np.abs(np.diff(v))
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8)]:
        m = (g[1:] > lo) & (g[1:] <= hi)
        print(f"  consecutive Si pairs, gap {lo}-{hi} h: n={m.sum():5d}  mean |dSi|={dv[m].mean():.3f}  "
              f"-> half of it is a rough per-sample noise level {dv[m].mean()/2:.3f}" if m.sum() else "")

    rows, preds = [], []
    t0 = time.time()
    VARIANTS = ["ref_ridge", "ref_hybrid", "hybrid+prev_state", "hybrid+slow48/72h",
                "delta_ridge", "delta_hybrid", "delta_hybrid+prev_state"]
    pooled = {v: [] for v in VARIANTS}
    for k in range(len(bounds) - 1):
        tr_end, va_end = bounds[k], bounds[k + 1]
        cleaner = SensorCleaner().fit(raw.loc[:dev.dt.iloc[tr_end - 1]], cfg.sensors)
        clean = cleaner.transform(raw)
        f2 = build_sensor_features(clean, raw, cfg, False, True)
        d_base = assemble(f2, dev, si, cfg, True)
        d_prev = prev_state_deltas(d_base, f2, si, cfg)
        d_slow = assemble(slow_features(f2, clean, cfg), dev, si, cfg, True)
        sets = {"ref_ridge": (d_base, lambda: make_model("ridge")),
                "ref_hybrid": (d_base, lambda: Hybrid()),
                "hybrid+prev_state": (d_prev, lambda: Hybrid()),
                "hybrid+slow48/72h": (d_slow, lambda: Hybrid()),
                "delta_ridge": (d_base, lambda: DeltaTarget(make_model("ridge"))),
                "delta_hybrid": (d_base, lambda: DeltaTarget(Hybrid())),
                "delta_hybrid+prev_state": (d_prev, lambda: DeltaTarget(Hybrid()))}
        for name, (d, mk) in sets.items():
            assert len(d) == n_dev
            cols = split_cols(d, cfg)
            tr, va = d.iloc[:tr_end], d.iloc[tr_end:va_end]
            p = mk().fit(tr[cols], tr["Si"]).predict(va[cols])
            r = evaluate(va["Si"], p)
            r.update(fold=k + 1, variant=name)
            rows.append(r)
            pooled[name].append(pd.DataFrame({"dt": va["dt"].values, "y": va["Si"].values, "p": p,
                                              "gap": va["hours_since_prev"].values}))
        print(f"fold {k+1} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/ideas_cv_folds.csv", index=False)
    s = df.groupby("variant")[["MAE", "R2", "within_0.10", "R2_normal", "MAE_normal"]].agg(["mean", "std"]).round(4)
    s.columns = ["_".join(c) for c in s.columns]
    s = s.loc[VARIANTS]
    s.to_csv(f"{OUT_DIR}/ideas_cv_summary.csv")
    print("\nCV (5 folds)\n", s[["MAE_mean", "MAE_std", "R2_mean", "R2_normal_mean", "within_0.10_mean"]].to_string())
    pf = df.pivot(index="fold", columns="variant", values="MAE")
    for v in VARIANTS[2:]:
        print(f"{v:26s} lower MAE than ref_hybrid in {(pf[v] < pf['ref_hybrid']).sum()} of 5 folds")

    # ---------------- error diagnostics for the reference hybrid ----------------
    a = pd.concat(pooled["ref_hybrid"]).reset_index(drop=True)
    a["err"] = a.y - a.p
    print("\nHybrid CV error by hours since previous Si result:")
    a["gap_bin"] = pd.cut(a.gap, [0, 1, 2, 3, 5, 10, 1e9], labels=["<1h", "1-2h", "2-3h", "3-5h", "5-10h", ">10h"])
    print(a.groupby("gap_bin", observed=True).agg(n=("err", "size"), MAE=("err", lambda x: x.abs().mean())).round(4).to_string())
    print("lag-1 autocorrelation of hybrid residuals:", round(a.err.autocorr(1), 3),
          " lag-2:", round(a.err.autocorr(2), 3))
    print("share of total absolute error from the 5 % largest errors:",
          round(float(a.err.abs().nlargest(int(len(a) * 0.05)).sum() / a.err.abs().sum()), 3))


if __name__ == "__main__":
    main()
