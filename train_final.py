"""
train_final.py - trains the deployment model on ALL available data and saves it with a model card.

  python train_final.py                    # writes model/si_pipeline_hybrid.pkl and model/model_card.json
  python train_final.py --demo-only        # only (re)build the held-out demo model, model/si_pipeline_demo.pkl

It also writes model/si_pipeline_demo.pkl: the SAME recipe fitted on the development period only (the first 85 % of
samples). The interface uses it to replay the held-out test period, so what a demo shows is out-of-sample.

Recipe (fixed by the experiments, not tuned here): Ridge + XGBoost-on-residuals hybrid, sensors + rolling features
+ the 3 previous Si results, 90 % split-conformal intervals calibrated on the last 15 % of samples, then refitted
on all samples. No held-out data remains after this step, so the accuracy to quote for the model is the
cross-validation and test-period result in results/consolidated_results.csv.
"""
import json
import os
import platform
import datetime as dt

import pandas as pd
import sklearn
import xgboost

from run_experiments import DATA_FILE
try:
    from si_pipeline import require_version
except ImportError:
    raise SystemExit("Your si_pipeline.py is out of date: copy the latest si_pipeline.py from the project folder over yours.")
require_version("1.3")
from si_pipeline import SiPipeline

MODEL_PATH = "model/si_pipeline_hybrid.pkl"
CARD_PATH = "model/model_card.json"
DEMO_PATH = "model/si_pipeline_demo.pkl"
DEMO_CARD_PATH = "model/model_card_demo.json"


def train_demo(sensor, si):
    """Same recipe, fitted on the development period only, so the test period stays unseen by this model."""
    from run_experiments import TEST_FRACTION
    from run_extra import load_universe
    _, _, _, samples = load_universe()
    n_dev = int(len(samples) * (1 - TEST_FRACTION))
    dev_end, test_start, test_end = samples.dt.iloc[n_dev - 1], samples.dt.iloc[n_dev], samples.dt.iloc[-1]
    t_s, t_q = pd.to_datetime(sensor["dt"]), pd.to_datetime(si["dt"])
    pipe = SiPipeline(use_lags=False, use_roll=True, use_history=True, model="hybrid",
                      calibrate=True, alpha=0.10, refit_on_all=False)    # the exact recipe evaluated in the paper
    pipe.fit(sensor[t_s <= dev_end], si[t_q <= dev_end])
    pipe.save(DEMO_PATH)
    from si_pipeline import prepare_si_frame
    si_ok = prepare_si_frame(si, pipe.cfg)
    si_ok = si_ok[si_ok[pipe.cfg.time_col] <= dev_end][pipe.cfg.target]
    card = {"name": "Si hybrid pipeline, held-out demo copy (fitted on the development period only)",
            "typical_si_range": [round(float(si_ok.quantile(0.10)), 3), round(float(si_ok.quantile(0.90)), 3)],
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "training_range": [str(pipe.train_range_[0]), str(pipe.train_range_[1])],
            "held_out_period": [str(test_start), str(test_end)],
            "interval_level": 1 - pipe.alpha, "interval_half_width_wt_pct_si": round(pipe.q_, 4),
            "purpose": "replaying the held-out test period in the interface; not for live use"}
    json.dump(card, open(DEMO_CARD_PATH, "w"), indent=2)
    print("saved", DEMO_PATH, "trained through", pipe.train_range_[1], "held out", test_start, "to", test_end)


def main():
    import sys
    sensor = pd.read_excel(DATA_FILE, sheet_name="data")
    si = pd.read_excel(DATA_FILE, sheet_name="Si")
    os.makedirs("model", exist_ok=True)
    if "--demo-only" in sys.argv:
        train_demo(sensor, si)
        return
    pipe = SiPipeline(use_lags=False, use_roll=True, use_history=True, model="hybrid",
                      calibrate=True, alpha=0.10, refit_on_all=True)
    pipe.fit(sensor, si)
    os.makedirs("model", exist_ok=True)
    pipe.save(MODEL_PATH)
    from si_pipeline import prepare_si_frame
    si_ok = prepare_si_frame(si, pipe.cfg)[pipe.cfg.target]
    card = {
        "name": "Si hybrid pipeline (Ridge + XGBoost on residuals)",
        "typical_si_range": [round(float(si_ok.quantile(0.10)), 3), round(float(si_ok.quantile(0.90)), 3)],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "training_samples": int(pipe.n_train_),
        "training_range": [str(pipe.train_range_[0]), str(pipe.train_range_[1])],
        "calibration_samples": int(pipe.n_cal_),
        "interval_level": 1 - pipe.alpha,
        "interval_half_width_wt_pct_si": round(pipe.q_, 4),
        "features": pipe.feature_names_,
        "inputs": {"sensors": pipe.cfg.sensors, "needs_previous_si_results": pipe.cfg.n_history,
                   "needs_sensor_history_hours": pipe.cfg.warmup_hours},
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "xgboost": xgboost.__version__, "pandas": pd.__version__},
        "known_limits": [
            "trained on one furnace only; do not use on other furnaces without retraining",
            "intervals cover about 88 % in normal operation in cross-validation (86-87 % on the test period) but almost never during upsets (Si > 1.2)",
            "coverage is lower after long gaps (> 2.5 h) since the previous lab result",
            "sensor-only information has almost no forward-looking skill; the previous Si results drive the prediction",
        ],
    }
    cons = "results/consolidated_results.csv"
    if os.path.exists(cons):
        t = pd.read_csv(cons).set_index("model")
        row = t.loc["Sensors + prev Si - Ridge+XGB hybrid"]
        card["evaluation_from_experiments"] = {"cv_mae": float(row.CV_MAE), "cv_r2": float(row.CV_R2),
                                                "test_mae": float(row.Test_MAE), "test_r2": float(row.Test_R2),
                                                "source": cons}
    json.dump(card, open(CARD_PATH, "w"), indent=2)
    print(json.dumps(card, indent=2)[:1500])
    print("saved", MODEL_PATH, "and", CARD_PATH)
    train_demo(sensor, si)


if __name__ == "__main__":
    main()