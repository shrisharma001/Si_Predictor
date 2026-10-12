"""
run_uncertainty.py - calibrated prediction intervals and upset early warning (same protocol as before).

PART 1  Prediction intervals for the primary model (Ridge + XGBoost hybrid), nominal coverage 90 %
  conformal  : split conformal, one symmetric half-width from calibration residuals
  mondrian   : conformal per bin of (long gap since previous lab result) x (recent Si volatility tercile)
  cqr        : conformalised quantile regression (XGBoost 5 % / 95 % quantiles + conformal correction)
  Each fold: fit on the first 85 % of the training period, calibrate on the last 15 % (chronological),
  evaluate on the next block.  Final: same recipe, dev -> test, evaluated ONCE for all three methods.

PART 2  Early warning of upsets (Si > 1.2 at the next cast), evaluated where the previous result is still
  normal (<= 1.0), i.e. warning BEFORE the upset has started.  Compared with the hybrid's point prediction
  used as an alarm score.  The number of upset episodes is very small, so treat this part as exploratory.

  python run_uncertainty.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve
from xgboost import XGBClassifier, XGBRegressor

from run_experiments import CV_BOUNDS, OUT_DIR, TEST_FRACTION, split_cols
from run_extra import load_universe
from run_peaks import Hybrid
from si_pipeline import Config, SensorCleaner, assemble, build_sensor_features

ALPHA = 0.10
UPSET = 1.2
GAP_H = 2.5
cfg = Config(n_history=3)


def cq(scores, alpha):
    s = np.sort(np.asarray(scores))
    k = min(int(np.ceil((len(s) + 1) * (1 - alpha))), len(s))
    return s[k - 1]


def quantile_model(alpha, seed=42):
    return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha, n_estimators=300, learning_rate=0.05,
                        max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                        random_state=seed, n_jobs=-1)


def bins(d, vol_edges=None):
    gap = (d["hours_since_prev"].values > GAP_H).astype(int)
    vol = np.abs(d["si_prev1"].values - d["si_prev2"].values)
    if vol_edges is None:
        vol_edges = np.quantile(vol, [1 / 3, 2 / 3])
    vb = np.digitize(vol, vol_edges)
    return gap * 3 + vb, vol_edges


def interval_metrics(y, lo, hi, d):
    cov = (y >= lo) & (y <= hi)
    width = hi - lo
    score = width + (2 / ALPHA) * np.clip(lo - y, 0, None) + (2 / ALPHA) * np.clip(y - hi, 0, None)
    up = y > UPSET
    lg = d["hours_since_prev"].values > GAP_H
    return {"coverage": cov.mean(), "width": width.mean(), "interval_score": score.mean(),
            "cov_normal": cov[~up].mean(), "cov_upset": cov[up].mean() if up.any() else np.nan, "n_upset": int(up.sum()),
            "cov_long_gap": cov[lg].mean() if lg.any() else np.nan, "n_long_gap": int(lg.sum()), "n": len(y)}


def intervals_for(tr, va, cols):
    """Return {method: (lo, hi)} for the validation block."""
    cut = int(len(tr) * 0.85)
    fit, cal = tr.iloc[:cut], tr.iloc[cut:]
    m = Hybrid().fit(fit[cols], fit.Si)
    p_cal, p_va = m.predict(cal[cols]), m.predict(va[cols])
    out = {}
    # 1) plain split conformal
    q = cq(np.abs(cal.Si.values - p_cal), ALPHA)
    out["conformal"] = (p_va - q, p_va + q)
    # 2) Mondrian by gap x volatility
    b_cal, edges = bins(cal)
    b_va, _ = bins(va, edges)
    res = np.abs(cal.Si.values - p_cal)
    qb = {b: (cq(res[b_cal == b], ALPHA) if (b_cal == b).sum() >= 40 else q) for b in range(6)}
    qv = np.array([qb[b] for b in b_va])
    out["mondrian"] = (p_va - qv, p_va + qv)
    # 3) conformalised quantile regression
    lo_m = quantile_model(ALPHA / 2).fit(fit[cols], fit.Si)
    hi_m = quantile_model(1 - ALPHA / 2).fit(fit[cols], fit.Si)
    lo_c, hi_c = lo_m.predict(cal[cols]), hi_m.predict(cal[cols])
    s = np.maximum(lo_c - cal.Si.values, cal.Si.values - hi_c)
    qc = cq(s, ALPHA)
    lo_v, hi_v = lo_m.predict(va[cols]), hi_m.predict(va[cols])
    out["cqr"] = (np.minimum(lo_v, hi_v) - qc, np.maximum(lo_v, hi_v) + qc)
    return out, p_va


def tpr_at_fpr(y, score, fpr_target=0.05):
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.interp(fpr_target, fpr, tpr))


def event_scores(tr, va, cols, p_va):
    """Scores for 'next Si > 1.2' on rows whose previous result is still normal (<= 1.0)."""
    ytr = (tr.Si.values > UPSET).astype(int)
    if ytr.sum() < 5:
        return None
    spw = min((len(ytr) - ytr.sum()) / ytr.sum(), 50.0)
    clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                        min_child_weight=2, reg_lambda=5.0, scale_pos_weight=spw, objective="binary:logistic",
                        random_state=42, n_jobs=-1).fit(tr[cols], ytr)
    prob = clf.predict_proba(va[cols])[:, 1]
    keep = va["si_prev1"].values <= 1.0
    return pd.DataFrame({"y": (va.Si.values > UPSET).astype(int)[keep], "classifier": prob[keep],
                         "hybrid_pred": p_va[keep], "persistence": va["si_prev1"].values[keep]})


def main():
    _, raw, si, samples = load_universe()
    n = len(samples)
    n_dev = int(n * (1 - TEST_FRACTION))
    dev = samples.iloc[:n_dev]
    bounds = [int(n_dev * b) for b in CV_BOUNDS]
    rows, ev_parts, t0 = [], [], time.time()
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[a - 1]], cfg.sensors)
        d = assemble(build_sensor_features(cl.transform(raw), raw, cfg, False, True), dev, si, cfg, True)
        cols = split_cols(d, cfg)
        tr, va = d.iloc[:a], d.iloc[a:b]
        iv, p_va = intervals_for(tr, va, cols)
        for meth, (lo, hi) in iv.items():
            r = interval_metrics(va.Si.values, lo, hi, va)
            r.update(fold=k + 1, method=meth)
            rows.append(r)
        e = event_scores(tr, va, cols, p_va)
        if e is not None:
            e["fold"] = k + 1
            ev_parts.append(e)
        print(f"fold {k+1} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/intervals_cv_folds.csv", index=False)
    s = df.groupby("method")[["coverage", "width", "interval_score", "cov_normal", "cov_upset", "cov_long_gap"]].agg(["mean", "std"]).round(3)
    s.columns = ["_".join(c) for c in s.columns]
    s.to_csv(f"{OUT_DIR}/intervals_cv_summary.csv")
    print(f"\nINTERVALS, CV (nominal coverage {1-ALPHA:.0%})\n", s.to_string())
    print("pooled upset samples (CV):", int(df[df.method == 'conformal'].n_upset.sum()),
          "| pooled long-gap samples:", int(df[df.method == 'conformal'].n_long_gap.sum()))

    # ---- final: dev -> test, all three methods, evaluated once ----
    cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[-1]], cfg.sensors)
    d = assemble(build_sensor_features(cl.transform(raw), raw, cfg, False, True), samples, si, cfg, True)
    cols = split_cols(d, cfg)
    tr, te = d.iloc[:n_dev], d.iloc[n_dev:]
    iv, p_te = intervals_for(tr, te, cols)
    out = []
    for meth, (lo, hi) in iv.items():
        r = interval_metrics(te.Si.values, lo, hi, te)
        r["method"] = meth
        out.append(r)
    t = pd.DataFrame(out).set_index("method").round(3)
    t.to_csv(f"{OUT_DIR}/intervals_test_results.csv")
    print(f"\nINTERVALS, TEST (nominal {1-ALPHA:.0%})\n", t.to_string())
    pr = te[["dt", "Si"]].copy()
    pr["pred"] = p_te
    for meth, (lo, hi) in iv.items():
        pr[f"{meth}_lo"], pr[f"{meth}_hi"] = lo, hi
    pr.to_csv(f"{OUT_DIR}/intervals_test_predictions.csv", index=False)

    # ---- early warning (exploratory) ----
    if ev_parts:
        ev = pd.concat(ev_parts)
        n_pos, n_all = int(ev.y.sum()), len(ev)
        print(f"\nEARLY WARNING (CV, rows with previous Si <= 1.0): {n_pos} upset onsets in {n_all} rows "
              f"(base rate {n_pos/n_all:.4f}); onsets per fold: {ev.groupby('fold').y.sum().to_dict()}")
        res = []
        for name in ["classifier", "hybrid_pred", "persistence"]:
            res.append({"score": name, "average_precision": average_precision_score(ev.y, ev[name]),
                        "recall_at_5pct_false_alarms": tpr_at_fpr(ev.y.values, ev[name].values)})
        e = pd.DataFrame(res).round(3)
        e.loc[len(e)] = ["(random guess: AP = base rate)", round(n_pos / n_all, 4), 0.05]
        e["n_onsets_in_cv"] = n_pos
        e["verdict"] = "NOT EVALUABLE: too few upset onsets (need >= 10 independent events)" if n_pos < 10 else "ok"
        if n_pos < 10:
            print(f"WARNING: only {n_pos} upset onset(s) in the CV data; the numbers below carry no statistical weight.")
        e.to_csv(f"{OUT_DIR}/early_warning_cv.csv", index=False)
        print(e.to_string(index=False))


if __name__ == "__main__":
    main()
