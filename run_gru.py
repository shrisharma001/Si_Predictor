"""
run_gru.py - recurrent baseline under the SAME protocol as the other models.

Input per Si sample
  * sequence : the last 24 hourly (cleaned) sensor rows up to the sample time  (24 x 28)
  * static   : 3 previous Si results, their mean, hours since the last result
Architecture : GRU(32) over the sequence + small MLP on the static vector, plus a LINEAR skip
               connection from the static vector (so the network can extrapolate like Ridge).
Training     : Huber loss, Adam, early stopping on the last 15 % (chronologically) of the training set,
               prediction = mean of N_SEEDS independently seeded networks.

  python run_gru.py                (needs: pip install tensorflow-cpu)
"""
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


def fit_predict(tr, te):
    (Xs, Xt, y), (Vs, Vt, _) = tr, te
    ms, ss = Xs.reshape(-1, Xs.shape[-1]).mean(0), Xs.reshape(-1, Xs.shape[-1]).std(0) + 1e-6
    mt, st = Xt.mean(0), Xt.std(0) + 1e-6
    sc = lambda s, t: [((s - ms) / ss).astype("float32"), ((t - mt) / st).astype("float32")]
    A, B = sc(Xs, Xt), sc(Vs, Vt)
    cut = int(len(y) * 0.85)
    preds = []
    for seed in range(N_SEEDS):
        m = make_net(Xs.shape[-1], Xt.shape[1], seed)
        m.fit([A[0][:cut], A[1][:cut]], y[:cut], validation_data=([A[0][cut:], A[1][cut:]], y[cut:]),
              epochs=80, batch_size=128, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
        preds.append(m.predict(B, verbose=0).ravel())
    return np.mean(preds, axis=0)


def main():
    _, raw, si, samples = load_universe()
    n = len(samples)
    n_dev = int(n * (1 - TEST_FRACTION))
    dev = samples.iloc[:n_dev]
    bounds = [int(n_dev * b) for b in CV_BOUNDS]
    rows, t0 = [], time.time()
    for k in range(len(bounds) - 1):
        tr_end, va_end = bounds[k], bounds[k + 1]
        cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[tr_end - 1]], cfg.sensors)
        clean = cl.transform(raw)
        seq, stat, y = build_arrays(clean, raw, dev, si)
        p = fit_predict((seq[:tr_end], stat[:tr_end], y[:tr_end]), (seq[tr_end:va_end], stat[tr_end:va_end], None))
        r = evaluate(y[tr_end:va_end], p)
        r.update(fold=k + 1, model="gru")
        rows.append(r)
        print(f"fold {k+1}: MAE={r['MAE']:.4f} R2={r['R2']:.3f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/gru_cv_folds.csv", index=False)
    print("\nCV mean:", df[["MAE", "R2", "within_0.10", "R2_normal", "MAE_normal"]].mean().round(4).to_dict(),
          "\nCV std :", df[["MAE", "R2"]].std().round(4).to_dict())

    # one-shot test evaluation
    cl = SensorCleaner().fit(raw.loc[:dev.dt.iloc[-1]], cfg.sensors)
    seq, stat, y = build_arrays(cl.transform(raw), raw, samples, si)
    p = fit_predict((seq[:n_dev], stat[:n_dev], y[:n_dev]), (seq[n_dev:], stat[n_dev:], None))
    r = evaluate(y[n_dev:], p)
    print("TEST:", {k: round(v, 4) for k, v in r.items()})
    pd.DataFrame([r]).to_csv(f"{OUT_DIR}/gru_test_results.csv", index=False)
    out = samples.iloc[n_dev:][["dt", "Si"]].copy()
    out["gru"] = p
    out.to_csv(f"{OUT_DIR}/gru_test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
