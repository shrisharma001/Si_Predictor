"""
run_gru.py - recurrent baseline under the SAME protocol as the other models, repeated over several runs.

Input per Si sample
  * sequence : the last 24 hourly (cleaned) sensor rows up to the sample time  (24 x 28)
  * static   : 3 previous Si results, their mean, hours since the last result
Architecture : GRU(32) over the sequence + small MLP on the static vector, plus a LINEAR skip
               connection from the static vector (so the network can extrapolate like Ridge).
Training     : Huber loss, Adam, early stopping on the last 15 % (chronologically) of the training set.

One RUN  = an ensemble of N_SEEDS independently seeded networks (their predictions are averaged).
The script performs N_RUNS independent runs (different seeds each) so the run-to-run spread of the
GRU is measured instead of guessed.  TensorFlow is not fully deterministic, so numbers still differ a
little between machines; always report the mean over runs.

Reported GRU numbers   : mean over runs of each run's metrics (typical performance).
gru_test_predictions   : predictions averaged over runs (used only for the paired bootstrap).

Usage
  python run_gru.py                        # 5 runs x 3 seeds, full CV + test  (slow: roughly 5 x 7 min)
  python run_gru.py --runs 5 --test-only   # skip CV, only the held-out test (much faster)
  python run_gru.py --runs 3 --tag _quick  # write gru_quick_*.csv instead of overwriting gru_*.csv
"""
import argparse
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from run_experiments import CV_BOUNDS, OUT_DIR, TEST_FRACTION
from run_extra import load_universe
from si_pipeline import Config, SensorCleaner, evaluate, history_features

L = 24
N_SEEDS = 3
cfg = Config(n_history=3)


def build_arrays(clean_df, raw, samples, si):
    arr = np.hstack([clean_df[cfg.sensors].values,
                     (pd.to_numeric(raw[cfg.blast_off_col], errors="coerce") <= 0).astype(float).values[:, None]])
    t = clean_df.index.values
    pos = np.searchsorted(t, samples.dt.values, side="right") - 1
    seq = np.stack([arr[p - L + 1:p + 1] for p in pos]).astype("float32")
    stat = history_features(samples.dt, si.dt, si.Si.values, cfg.n_history).values.astype("float32")
    return seq, stat, samples["Si"].values.astype("float32")


def make_net(F, S, seed):
    keras.utils.set_random_seed(seed)
    a, b = keras.Input((L, F)), keras.Input((S,))
    g = keras.layers.Dropout(0.2)(keras.layers.GRU(32)(a))
    s = keras.layers.Dense(16, activation="relu")(b)
    z = keras.layers.Dense(16, activation="relu")(keras.layers.Concatenate()([g, s]))
    out = keras.layers.Add()([keras.layers.Dense(1)(z), keras.layers.Dense(1)(b)])
    m = keras.Model([a, b], out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=0.15))
    return m


def fit_predict(tr, te, seed_base):
    (Xs, Xt, y), (Vs, Vt, _) = tr, te
    ms, ss = Xs.reshape(-1, Xs.shape[-1]).mean(0), Xs.reshape(-1, Xs.shape[-1]).std(0) + 1e-6
    mt, st = Xt.mean(0), Xt.std(0) + 1e-6
    sc = lambda s, t: [((s - ms) / ss).astype("float32"), ((t - mt) / st).astype("float32")]
    A, B = sc(Xs, Xt), sc(Vs, Vt)
    cut = int(len(y) * 0.85)
    preds = []
    for k in range(N_SEEDS):
        m = make_net(Xs.shape[-1], Xt.shape[1], seed_base + k)
        m.fit([A[0][:cut], A[1][:cut]], y[:cut], validation_data=([A[0][cut:], A[1][cut:]], y[cut:]),
              epochs=80, batch_size=128, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
        preds.append(m.predict(B, verbose=0).ravel())
    return np.mean(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5, help="number of independent runs")
    ap.add_argument("--test-only", action="store_true", help="skip cross-validation")
    ap.add_argument("--max-folds", type=int, default=None, help="limit CV folds (debugging)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _quick")
    a = ap.parse_args()

    _, raw, si, samples = load_universe()
    n = len(samples)
    n_dev = int(n * (1 - TEST_FRACTION))
    dev = samples.iloc[:n_dev]
    bounds = [int(n_dev * b) for b in CV_BOUNDS]

    # ---- prepare all arrays once ------------------------------------------------------------
    folds = []
    if not a.test_only:
        for k in list(range(len(bounds) - 1))[: a.max_folds]:
            tr_end, va_end = bounds[k], bounds[k + 1]
            cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[tr_end - 1]], cfg.sensors)
            seq, stat, y = build_arrays(cl.transform(raw), raw, dev, si)
            folds.append((k + 1, (seq[:tr_end], stat[:tr_end], y[:tr_end]),
                          (seq[tr_end:va_end], stat[tr_end:va_end], None), y[tr_end:va_end]))
    cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[-1]], cfg.sensors)
    seq, stat, y = build_arrays(cl.transform(raw), raw, samples, si)
    test_tr = (seq[:n_dev], stat[:n_dev], y[:n_dev])
    test_te = (seq[n_dev:], stat[n_dev:], None)
    y_test = y[n_dev:]

    # ---- runs --------------------------------------------------------------------------------
    cv_rows, test_rows, test_preds = [], [], []
    t0 = time.time()
    for r in range(a.runs):
        base = 1000 * (r + 1)
        for fold, tr, va, yv in folds:
            p = fit_predict(tr, va, base)
            m = evaluate(yv, p)
            m.update(run=r + 1, fold=fold, model="gru")
            cv_rows.append(m)
        p = fit_predict(test_tr, test_te, base)
        m = evaluate(y_test, p)
        m.update(run=r + 1, model="gru")
        test_rows.append(m)
        test_preds.append(p)
        msg = f"run {r+1}/{a.runs}: TEST MAE={m['MAE']:.4f} R2={m['R2']:.3f} within0.1={m['within_0.10']:.3f}"
        if folds:
            cvr = pd.DataFrame([x for x in cv_rows if x["run"] == r + 1])
            msg += f" | CV MAE={cvr.MAE.mean():.4f} R2={cvr.R2.mean():.3f}"
        print(msg + f"  ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- save --------------------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = ["MAE", "RMSE", "R2", "within_0.05", "within_0.10", "R2_normal", "MAE_normal"]
    tdf = pd.DataFrame(test_rows)
    tdf.to_csv(f"{OUT_DIR}/gru{a.tag}_test_runs.csv", index=False)
    summary = {k: tdf[k].mean() for k in metrics}
    summary.update({f"{k}_std": tdf[k].std() for k in ["MAE", "R2", "within_0.10", "R2_normal"]})
    summary.update(n=int(tdf["n"].iloc[0]), runs=a.runs, seeds_per_run=N_SEEDS)
    pd.DataFrame([summary]).to_csv(f"{OUT_DIR}/gru{a.tag}_test_results.csv", index=False)
    out = samples.iloc[n_dev:][["dt", "Si"]].copy()
    out["gru"] = np.mean(test_preds, axis=0)
    out.to_csv(f"{OUT_DIR}/gru{a.tag}_test_predictions.csv", index=False)
    print("\nTEST over runs (mean +- std):")
    for k in ["MAE", "R2", "within_0.10", "R2_normal"]:
        print(f"  {k:12s} {tdf[k].mean():.4f} +- {tdf[k].std():.4f}")

    if cv_rows:
        cdf = pd.DataFrame(cv_rows)
        cdf.to_csv(f"{OUT_DIR}/gru{a.tag}_cv_runs.csv", index=False)
        per_fold = cdf.groupby("fold")[metrics + ["n"]].mean().reset_index()   # mean over runs, per fold
        per_fold["model"] = "gru"
        per_fold.to_csv(f"{OUT_DIR}/gru{a.tag}_cv_folds.csv", index=False)
        per_run = cdf.groupby("run")[["MAE", "R2"]].mean()
        print("\nCV over runs (mean of fold means +- std across runs):")
        print(f"  MAE {per_run.MAE.mean():.4f} +- {per_run.MAE.std():.4f}   R2 {per_run.R2.mean():.3f} +- {per_run.R2.std():.3f}")


if __name__ == "__main__":
    main()