# Blast Furnace Hot-Metal Silicon (Si) Prediction

Predicts the silicon content of hot metal (Si, wt%) at each cast from hourly blast-furnace
process data and the previous Si lab results. The goal is a reusable pipeline (cleaning,
features, models, evaluation), served by a Flask API with an operator interface for a local demo.

## Data

- File: `data/1_blast_furnace_data_first_dataset.xlsx` (not committed; see `.gitignore`)
  - sheet `data`: 29,602 hourly rows, 27 process variables (`Fb, Ph, Pc, Tc, Fo, dP, dPu, dPl, Pt, Th,
    CO2, H2, Tt1-Tt4, Tp1-Tp10, R`) plus timestamp `dt`
  - sheet `Si`: 16,589 irregularly timed Si lab results
  - sheet `symbols`: variable descriptions
- Source: *Process data collected by blast-furnace sensors*, Mendeley Data,
  DOI 10.17632/6d7jbc7tb5.1
  <!-- (Version 1, 15 Sep 2025; the description matches: 26 sensors plus a computed ore-to-coke ratio, a raw file and a validated, class-labelled file). Check the licence on the page before publishing. -->
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
| `run_uncertainty.py` | 90 % prediction intervals (conformal, Mondrian, CQR) with CV and one-shot test; exploratory upset early warning (too few events to evaluate) |
| `run_learning_curve.py` | MAE vs amount of recent training data |
| `run_thermal.py` | CV-only test of hot-blast-temperature (`Th`) based thermal-state features; no gain |
| `run_domain.py` | CV-only test of process-knowledge features (oxygen share, permeability index, temperature-profile summaries); no gain |
| `run_ideas.py`, `run_ideas2.py` | CV-only tests of further ideas (state-change features, delta target, slow features, sliding window, error feedback, blend); none beat the hybrid |
| `run_split_comparison.py` | Shows how random / blocked / chronological splits change the reported accuracy (development period only) |
| `run_gru.py` | GRU baseline (TensorFlow) under the same protocol; several independent runs, reports mean and spread |
| `train_final.py` | Trains the live model on all data (hybrid + 90 % conformal intervals) into `model/si_pipeline_hybrid.pkl` and `model/model_card.json`, and a held-out demo copy trained on the development period only into `model/si_pipeline_demo.pkl` and `model/model_card_demo.json` (`--demo-only` builds just the demo copy) |
| `test_pipeline.py` | 12 checks: the pipeline reproduces the experiments (MAE, interval coverage, width), input validation, warnings, save/load, coverage with and without refitting |
| `app.py` | Flask API and operator interface: `/` (interface), `/api/predict` (JSON), `/api/predict/upload` (CSV/Excel files), `/api/demo/info` and `/api/demo/replay` (held-out replay), `/health`, `/api/model`, `/api/example`, `/console` (raw JSON console) |
| `static/` | The operator interface: `index.html`, `app.css`, `app.js` (no external libraries; fonts fall back to system fonts offline) |
| `make_sample_request.py` | Writes `examples/sample_request.json` (72 hourly sensor rows, 10 previous Si results, a query time) |
| `test_api.py` | 56 checks: JSON API, file uploads (CSV, Excel, mixed timestamp styles), every validation error, data-quality handling, the demo replay, the interface files, and three leakage checks (future lab results, future sensor rows, replay consistency) |
| `Dockerfile`, `docker-entrypoint.sh`, `docker-compose.yml`, `.dockerignore` | Container for the API and interface; the start-up script has `train`, `serve` and `test` commands (see "Docker") |
| `requirements-docker.txt` | Exact library versions for the image (Python 3.12, CPU-only XGBoost) |
| `docker_smoke_test.sh` | Builds the image and checks the whole container workflow on your machine |
| `requirements-serve.txt` | Minimal serving requirements without exact versions, for running the API outside Docker |
| `predict_example.py` | How the saved pipeline is called (`predict_frame`); the API and UI will use the same call |
| `run_sensitivity.py` | One-at-a-time hyperparameter sensitivity of the hybrid (CV only; 15 variants changed MAE by -0.3 % to +0.6 %) |
| `make_paper_figures.py` | Redraws all paper figures at single-column size into `paper/figs/` and writes `results/test_hybrid_predictions.csv`, `results/hybrid_feature_importance.csv` |
| `export_data_facts.py` | Writes `results/data_facts.csv` and `results/cleaner_report.csv` (dataset counts, sampling gaps, upset counts, detected sensor faults) |
| `build_paper_numbers.py` | Writes `paper/paper_numbers.md`: every number the paper may quote, generated from `results/` |
| `consolidate.py` | Builds `results/consolidated_results.csv` and the GRU significance tests |
| `train_model.py`, `train_model_lag.py` | Earlier XGBoost baselines |
| `templates/` | Old single-row form for the earlier model; no longer used (the interface is in `static/`) and can be deleted |
| `analyze_*.py` | Earlier exploratory analyses (lags, split shift, outliers) |

## Run order
Later steps read files written by earlier ones, so keep this order. **First copy the latest `si_pipeline.py`**: the deployment scripts check its
version and stop with a message if it is older than they need.

```bash
# 1. Experiments (write to results/)
python run_experiments.py --stage cv
python run_experiments.py --stage final --configs "F0:mean,F0:xgb_l2,F1:xgb_l2,F0+H:persistence,F0+H:xgb_l2,F2+H:ridge,F3+H:xgb_l2,F3+H:xgb_huber"
python run_extra.py --stage history
python run_extra.py --stage classify
python run_extra.py --stage analysis
python run_peaks.py
python run_split_comparison.py
python run_uncertainty.py
python run_learning_curve.py
python run_sensitivity.py
python run_ideas.py && python run_ideas2.py && python run_domain.py && python run_thermal.py     # negative results
python run_gru.py                  # needs tensorflow-cpu; 5 runs, about an hour (add --test-only for a quick check)

# 2. Paper material
python consolidate.py              # needs the GRU results
python export_data_facts.py
python make_paper_figures.py
python build_paper_numbers.py      # last; lists any missing result file and the script that creates it

# 3. Deployment
python train_final.py              # live model + held-out demo model (both needed by the interface)
python test_pipeline.py            # 12 checks, a few minutes
python make_sample_request.py
python test_api.py                 # 56 checks
python app.py                      # open http://127.0.0.1:5000
```

## Troubleshooting
| Message | Cause and fix |
|---|---|
| `si_pipeline.py is out of date` / `unexpected keyword argument 'calibrate'` | Your `si_pipeline.py` is older than the scripts. Copy the latest one into the project folder. |
| `FileNotFoundError: results/<name>.csv` | The script that creates it has not been run yet. `build_paper_numbers.py` prints the script for each missing file. |
| `Model file not found: model/si_pipeline_hybrid.pkl` | Run `python train_final.py` (the model is gitignored, so it is not in the repository). |
| Pickle or version errors when loading the model | Retrain with `python train_final.py` in the same environment that will serve the API. |
| `Replay is not available` in the interface | The demo model or the dataset is missing: run `python train_final.py --demo-only` and keep the dataset in `data/`. |
| The replay button says `Loading replay data…` for about 10 s after start-up | Normal: the server reads the 7 MB dataset once in the background. Open the page a little before the demo. |
| An uploaded Excel file is slow | Excel parsing takes several seconds for a full year of data; CSV files are much faster. |

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
| Sensors + prev. Si, GRU (mean of 5 runs x 3 seeds) | 0.079 | -0.17 | 0.082 | 0.68 | 0.739 |

Full table: `results/consolidated_results.csv`. The GRU row is the mean of 5 independent runs
(test MAE 0.0823 +- 0.0009, R2 0.684 +- 0.009; CV R2 varies from -0.77 to 0.17 between runs).

## Findings and caveats

- Sensors alone have almost no forward-looking skill on time-split data.
- The previous Si lab results are the key input; more than 3 add almost nothing.
- The split method matters more than the model: for sensors-only Random Forest, R² is 0.52 with a random split,
  0.38 / -0.12 / -0.03 with 1-day / 7-day / 30-day blocks, and 0.02 with a chronological split (development period only).
  MAE moves much less (0.071 to 0.079), so R² on shuffled splits is the most misleading number.
- Nominal 90 % intervals cover 88 % in CV and 86-87 % on the test set, with width about 0.29 (about +-0.14 Si). Coverage is
  about 88 % in normal operation but poor in upsets (1 % CV, 19 % test for conformal; CQR reaches 51 % on test), and low
  after long gaps between lab results (0.74 to 0.82).
- Learning curve: MAE keeps falling with more training data (0.075 at 2,000 samples to 0.067 at 10,000, no plateau).
- Upset early warning cannot be evaluated: the whole series contains only 8 upset onsets (3 development, 5 test).
- Models beat persistence significantly (paired block bootstrap), mainly in normal operation.
- All models underestimate the large upsets (Si > 1.2, about 2 % of samples). Test R² is strongly
  affected by the Jan-Feb 2016 upsets, so also look at the "normal-range" metrics.
- Ridge vs XGBoost and hybrid vs Ridge differences are not statistically significant on the test set.
- The GRU is unstable in CV (mean R2 -0.17 over 5 runs; one fold is about -1) and, on the test set, is
  statistically tied with the hybrid (0.0823 vs 0.0812, 95 % CI of the difference -0.0022 to +0.0026).
  The hybrid has lower CV MAE in every fold, so it is the primary model; the GRU is a comparator.
- "Normal-range" metrics select samples by true Si (0.2 to 1.2).
- The test set was used twice in total (main configurations, then peak and GRU experiments),
  plus once for the band-classification analysis. Re-running the GRU repeatedly to find a better test
  number would be test-set tuning: fix the number of runs in advance and report their mean.
- Results are for a single furnace; different furnaces need their own trained model.
- `2_blast_furnace_data_second_dataset.xlsx` is a labelled excerpt (200 hours, 127 Si samples, 8 furnace states of 25 hours each) of the first dataset, not a second furnace and not extra training data.

## Files in `results/`

| File | Written by | Content |
|---|---|---|
| `cv_fold_results.csv`, `cv_summary.csv` | `run_experiments.py --stage cv` | CV metrics per fold, and mean/std per feature set and model |
| `final_test_results.csv`, `final_test_predictions.csv` | `--stage final` | One-shot test metrics and predictions for the pre-specified configurations |
| `importance_*.csv` | `--stage final` | XGBoost feature importances (`F0HH` means `F0+H`: the `+` is replaced by `H` in file names) |
| `cv_history_folds.csv`, `cv_history_summary.csv` | `run_extra.py --stage history` | Number of previous Si results (1, 3, 5, 10) and the MLP baseline |
| `cv_band_summary.csv`, `test_band_results.csv` | `--stage classify` | Low / normal / high Si-band classification |
| `bootstrap_tests.csv`, `error_by_regime.csv` | `--stage analysis` | Paired block-bootstrap tests and error by Si regime |
| `fig_test_timeseries.png`, `fig_scatter.png` | `run_extra.py --stage analysis` | Paper figures (300 dpi) |
| `fig_cv_mae.png` | `consolidate.py` | MAE of all 7 models, CV and test (`run_extra.py` writes an older 5-bar version first; `consolidate.py` overwrites it, so keep the run order) |
| `cv_peaks_folds.csv`, `cv_peaks_summary.csv`, `test_peaks_results.csv`, `test_peaks_predictions.csv` | `run_peaks.py` | Early-warning features, log target, hybrid |
| `gru_cv_folds.csv`, `gru_test_results.csv`, `gru_test_predictions.csv`, `gru_test_runs.csv`, `gru_cv_runs.csv` | `run_gru.py` | GRU baseline (means over runs, plus per-run values) |
| `intervals_cv_*.csv`, `intervals_test_*.csv`, `early_warning_cv.csv`, `fig_intervals_test.png` | `run_uncertainty.py` (`--figure-only` redraws the figure from saved predictions) | Prediction-interval coverage/width and the (non-evaluable) early-warning check |
| `learning_curve_*.csv`, `fig_learning_curve.png` | `run_learning_curve.py` | MAE vs training size |
| `thermal_cv_*.csv` | `run_thermal.py` | Th-based features (negative result) |
| `domain_cv_*.csv` | `run_domain.py` | CV results of the domain features (negative result) |
| `ideas_cv_*.csv`, `ideas2_cv_*.csv` | `run_ideas*.py` | CV results of the further ideas (negative results) |
| `split_comparison.csv`, `split_comparison_runs.csv`, `fig_split_leakage.png` | `run_split_comparison.py` | Accuracy under random, blocked and chronological splits |
| `consolidated_results.csv` | `consolidate.py` | Final comparison table across all models |

## What to commit

Commit the `.py` files, `templates/`, `requirements.txt`, `README.md`, and the small `results/`
CSVs and `fig_*.png`. Do not commit `data/`, `model/*.pkl`, `venv/` or `__pycache__/`.

## Paper materials (`paper/`)
- `paper_numbers.md` generated numbers with their source files; `references.bib` and `REFERENCES_CHECK.md` the verified bibliography and what is still unconfirmed; `paper_outline.md` titles, claims, section plan and draft Related Work, Data and Protocol text.

## Deployment pipeline
`SiPipeline(model="hybrid", calibrate=True)` bundles cleaning, features, the hybrid model and conformal intervals in one object. `predict_frame(sensor_window, query_times, si_history)` returns, per query, the prediction, a 90 % interval, a status (`ok` / `warning` / `no_prediction`) and plain-language messages (missing history, fault codes replaced, out-of-range inputs, stale previous Si result, upsets, blast off). It needs at least 24 hourly sensor rows before each query and 3 previous Si results. Train with `python train_final.py`, check with `python test_pipeline.py`. The `.pkl` is tied to the scikit-learn/xgboost versions that created it, so retrain rather than copy it between machines.

## Operator interface (local demo)
Run `python train_final.py` once, then `python app.py`, and open http://127.0.0.1:5000. Open it a few seconds before presenting, since
the replay data loads in the background at start-up.

- **Replay held-out casts.** Uses the demo model, which was trained only on data up to 11 October 2015, and replays casts from the
  held-out period (11 October 2015 to 11 May 2016) one by one on a strip chart: measured silicon in blue, predictions in red with the
  90 % interval. Each prediction uses only data from before its cast, so this is out-of-sample performance. Presets cover a normal week
  (March 2016) and the largest upset (January 2016); up to 200 casts per replay.
- **Predict from my files.** Upload a sensor file (`dt` plus the 27 variables, at least 24 hourly rows before the chosen time) and an Si
  file (`dt`, `Si`, at least 3 results), as CSV or Excel. A single workbook with a `data` and an `Si` sheet works for both fields.
  This uses the live model.
- The readout shows the prediction, its interval against the typical range (middle 80 % of past casts, 0.36 to 0.61 wt%), and the status
  with plain-language warnings. "Show the numbers" gives a table of everything on the chart.
- Seen in the replay: thermocouple `Tp4` reads about 3,300 °C for long stretches of 2016, which is a sensor failure. The cleaner replaces
  those readings, and the interface reports it as a warning.

## Docker
The container serves the API and interface with gunicorn. The dataset and the models are not part of the image: `./data` is
mounted read-only and the container **trains its own models into `./model_docker`** with the image's libraries, so the saved
pickles always match the library versions that load them. `./model` stays for `python app.py` on your computer, so the two
sets of models never overwrite each other.

```bash
docker compose build                   # once, a few minutes
docker compose run --rm si train       # once, about a minute: writes both models to ./model_docker
docker compose up                      # open http://localhost:8000 ; Ctrl+C to stop
./docker_smoke_test.sh                 # optional: builds, trains, tests and checks the running service
```
- If `id -u` is not 1000, prefix the commands with `HOST_UID=$(id -u) HOST_GID=$(id -g)` so the model files belong to you.
- `docker compose run --rm si test` runs the 56 API checks inside the container.
- The port is published on 127.0.0.1 only, so the demo is reachable from this computer only.
- `./model_docker` (container) and `./model` (your computer) are separate on purpose: each set is trained with the libraries that load it.
- What was tested: the Docker Hub registry was not reachable from the authoring environment, so the image itself was not built
  there. Everything inside it was run as the container would run it: Python 3.12 with the exact pinned libraries, only the files
  the Dockerfile copies, an unprivileged user, a read-only dataset, gunicorn with two workers, the health check, `docker stop`
  (clean exit in about a second), all three commands and their error messages, and `docker_smoke_test.sh` end to end. With these
  libraries the reproduction checks still pass (test MAE 0.0809 against 0.0812, coverage 0.870 against 0.868). Run
  `./docker_smoke_test.sh` once on your machine to confirm the build.

## Prediction API
Start: `python train_final.py` (once), `python make_sample_request.py` (once), then `python app.py` (development, http://127.0.0.1:5000) or
`gunicorn -w 2 -b 0.0.0.0:8000 app:app` (production, Linux/macOS). Test without a server: `python test_api.py`.

`POST /api/predict` with `Content-Type: application/json`:
```json
{
  "sensor_readings": [ {"dt": "2015-10-19T07:00:00", "Fb": 3300.0, "Ph": 3.1, "...": "all 27 variables"}, "... at least 24 hourly rows" ],
  "si_history":      [ {"dt": "2015-10-19T05:10:00", "Si": 0.48}, "... at least 3 results before the query" ],
  "query_time":      "2015-10-19T08:00:00"
}
```
- `query_time` is optional (default: the latest sensor reading); `query_times` (a list) asks for several predictions at once.
- Results stamped at or after a query time are ignored, so a request can never leak the future.
- Response: `status` (`ok`, `warning`, `partial`, `no_prediction`), and per query `prediction`, `lower`, `upper`, `interval_level` (0.9), `status`, `messages`.
- HTTP codes: 200 predictions made; 400 invalid input (the body names the problem); 413 body too large; 415 not JSON; 422 no prediction possible (for example fewer than 24 hourly rows or fewer than 3 previous results).
- Messages warn about: replaced fault readings, values outside the training range, a previous Si result older than 6 h, a previous result above 1.2 (upsets), blast off.
- Limits: 8 MB body, 5,000 sensor rows, 5,000 history rows, 500 query times.

Operating notes
- Debug mode is off unless `FLASK_DEBUG=1`; the default bind address is 127.0.0.1. Never run debug mode on a public address.
- There is no authentication or rate limiting in `app.py`: put the API behind a reverse proxy or gateway that provides them before exposing it publicly.
- The model file is a pickle: load only files you created. It is gitignored; decide where it lives (a release artifact, a storage bucket, or built on the server with `train_final.py`) and keep the library versions identical to the ones that trained it.
- `POST /api/predict/upload` takes multipart form fields `sensor_file`, `si_file` and optional `query_time`; files are trimmed to the 7 days before the query time, and rows after it are ignored.
- `POST /api/demo/replay` takes `{"start": "2016-03-10T00:00", "casts": 48}` and returns, per cast, the prediction, interval, measured value and persistence value, plus summary MAE and coverage.
- For the local demo, authentication is not needed. Before any cloud deployment (planned for next semester), add authentication, rate limiting and a decision on where model files are stored.