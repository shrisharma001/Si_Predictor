# Blast Furnace Hot-Metal Silicon (Si) Prediction

Predicts the silicon content of hot metal (Si, wt%) at each cast from hourly blast-furnace
process data and the previous Si lab results. The goal is a reusable pipeline (cleaning,
features, models, evaluation) that can later be served through a Flask UI.

## Data

- File: `data/1_blast_furnace_data_first_dataset.xlsx` (not committed; see `.gitignore`)
  - sheet `data`: 29,602 hourly rows, 27 process variables (`Fb, Ph, Pc, Tc, Fo, dP, dPu, dPl, Pt, Th,
    CO2, H2, Tt1-Tt4, Tp1-Tp10, R`) plus timestamp `dt`
  - sheet `Si`: 16,589 irregularly timed Si lab results
  - sheet `symbols`: variable descriptions
- Source: *Process data collected by blast-furnace sensors*, Mendeley Data,
  DOI 10.17632/6d7jbc7tb5.1. **TODO: confirm this matches the file used before citing.**
- Period: January 2013 to May 2016.

## Setup

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data model results
# place 1_blast_furnace_data_first_dataset.xlsx in data/
```

## Files

| File | Purpose |
|---|---|
| `si_pipeline.py` | Core module: sensor cleaner, lag/rolling features, previous-Si features, models, metrics, `SiPipeline` (one saveable object for training and inference) |
| `run_experiments.py` | Main protocol: time-series CV on the first 85 %, one-shot final test on the last 15 % |
| `run_extra.py` | History-length sweep + MLP, Si-band classification, paired block-bootstrap tests, error analysis, figures |
| `run_peaks.py` | Upset/peak experiments (early-warning features, log target, Ridge+XGBoost hybrid) |
| `run_gru.py` | GRU baseline (TensorFlow) under the same protocol |
| `consolidate.py` | Builds `results/consolidated_results.csv` and the GRU significance tests |
| `app.py`, `templates/` | Flask app (currently loads the older 27-feature model, not yet the new pipeline) |
| `train_model.py`, `train_model_lag.py` | Earlier XGBoost baselines |
| `analyze_*.py` | Earlier exploratory analyses (lags, split shift, outliers) |

## Run order

Later steps read files written by earlier ones, so keep this order.

```bash
python run_experiments.py --stage cv
python run_experiments.py --stage final --configs "F0:mean,F0:xgb_l2,F1:xgb_l2,F0+H:persistence,F0+H:xgb_l2,F2+H:ridge,F3+H:xgb_l2,F3+H:xgb_huber"
python run_extra.py --stage history
python run_extra.py --stage classify
python run_extra.py --stage analysis
python run_peaks.py
python run_gru.py          # needs tensorflow-cpu
python consolidate.py
```

Results land in `results/`. Seeds are fixed, so every model except the GRU reproduces exactly
(a second machine reproduced the Ridge, XGBoost and hybrid numbers to all printed digits).
TensorFlow is not fully deterministic, so GRU numbers differ between runs and machines; see below.

## Evaluation protocol

- All aligned Si samples are sorted by time; each is matched to the latest sensor row at or before it
  (backward, max 1 hour).
- **Test set = last 15 %**, never used for model or feature selection; evaluated once per
  pre-specified configuration.
- First 85 % = expanding-window time-series CV (5 folds). The sensor cleaner is re-fitted on each
  fold's training period only.
- Previous-Si features use strictly earlier lab results only.
- No random splits: random splits on this data leak neighbouring hours into the test set and inflate R².

## Results (3 previous Si results used)

| Model | CV MAE | CV R² | Test MAE | Test R² | Test within ±0.1 |
|---|---|---|---|---|---|
| Mean (no skill) | 0.096 | -0.05 | 0.114 | 0.00 | 0.648 |
| Sensors only (XGBoost) | 0.092 | 0.01 | 0.109 | 0.17 | 0.651 |
| Persistence (previous Si) | 0.078 | 0.10 | 0.100 | 0.60 | 0.643 |
| Sensors + prev. Si, XGBoost | 0.074 | 0.27 | 0.086 | 0.62 | 0.727 |
| Sensors + prev. Si, Ridge | 0.071 | 0.33 | 0.083 | 0.71 | 0.732 |
| Sensors + prev. Si, Ridge+XGBoost hybrid | **0.069** | **0.38** | 0.081 | 0.72 | 0.734 |
| Sensors + prev. Si, GRU (3 seeds, one run) | 0.074 | 0.06 | 0.081 | 0.70 | 0.742 |

Full table: `results/consolidated_results.csv`. The GRU row is from one run on the author's machine;
an earlier run elsewhere gave test MAE 0.079 and R² 0.72, so the GRU and the hybrid are effectively tied on test.

## Findings and caveats

- Sensors alone have almost no forward-looking skill on time-split data.
- The previous Si lab results are the key input; more than 3 add almost nothing.
- Models beat persistence significantly (paired block bootstrap), mainly in normal operation.
- All models underestimate the large upsets (Si > 1.2, about 2 % of samples). Test R² is strongly
  affected by the Jan-Feb 2016 upsets, so also look at the "normal-range" metrics.
- Ridge vs XGBoost and hybrid vs Ridge differences are not statistically significant on the test set.
- The GRU is unstable in CV (one fold R² of -0.96) and, on the test set, roughly tied with the hybrid
  (test MAE 0.079 to 0.081 across two runs, vs 0.081). Do not claim it is better; treat it as a comparator.
  For the paper, report the mean over several runs rather than a single run.
- "Normal-range" metrics select samples by true Si (0.2 to 1.2).
- The test set was used twice in total (main configurations, then peak and GRU experiments),
  plus once for the band-classification analysis. Re-running the GRU repeatedly to find a better test
  number would be test-set tuning: fix the number of runs in advance and report their mean.
- Results are for a single furnace; different furnaces need their own trained model.

## Files in `results/`

| File | Written by | Content |
|---|---|---|
| `cv_fold_results.csv`, `cv_summary.csv` | `run_experiments.py --stage cv` | CV metrics per fold, and mean/std per feature set and model |
| `final_test_results.csv`, `final_test_predictions.csv` | `--stage final` | One-shot test metrics and predictions for the pre-specified configurations |
| `importance_*.csv` | `--stage final` | XGBoost feature importances (`F0HH` means `F0+H`: the `+` is replaced by `H` in file names) |
| `cv_history_folds.csv`, `cv_history_summary.csv` | `run_extra.py --stage history` | Number of previous Si results (1, 3, 5, 10) and the MLP baseline |
| `cv_band_summary.csv`, `test_band_results.csv` | `--stage classify` | Low / normal / high Si-band classification |
| `bootstrap_tests.csv`, `error_by_regime.csv` | `--stage analysis` | Paired block-bootstrap tests and error by Si regime |
| `fig_test_timeseries.png`, `fig_scatter.png`, `fig_cv_mae.png` | `--stage analysis` / `consolidate.py` | Paper figures (300 dpi) |
| `cv_peaks_folds.csv`, `cv_peaks_summary.csv`, `test_peaks_results.csv`, `test_peaks_predictions.csv` | `run_peaks.py` | Early-warning features, log target, hybrid |
| `gru_cv_folds.csv`, `gru_test_results.csv`, `gru_test_predictions.csv` | `run_gru.py` | GRU baseline |
| `consolidated_results.csv` | `consolidate.py` | Final comparison table across all models |

## What to commit

Commit the `.py` files, `templates/`, `requirements.txt`, `README.md`, and the small `results/`
CSVs and `fig_*.png`. Do not commit `data/`, `model/*.pkl`, `venv/` or `__pycache__/`.