"""
export_data_facts.py - writes the dataset facts quoted in the paper to results/data_facts.csv
(and the sensor-cleaning report to results/cleaner_report.csv), so every such number has a source file.

  python export_data_facts.py
"""
import os

import numpy as np
import pandas as pd

from run_experiments import DATA_FILE, OUT_DIR, TEST_FRACTION
from run_extra import load_universe
from si_pipeline import Config, SensorCleaner, prepare_sensor_frame, prepare_si_frame

cfg = Config(n_history=3)
os.makedirs(OUT_DIR, exist_ok=True)
sheet = pd.read_excel(DATA_FILE, sheet_name="data")
si_raw = pd.read_excel(DATA_FILE, sheet_name="Si")
raw = prepare_sensor_frame(sheet, cfg)
si = prepare_si_frame(si_raw, cfg)
_, raw2, si2, samples = load_universe()
n = len(samples)
n_dev = int(n * (1 - TEST_FRACTION))
dev, test = samples.iloc[:n_dev], samples.iloc[n_dev:]

rows = []
def add(key, value, note=""):
    rows.append({"fact": key, "value": value, "note": note})

add("sensor_rows", len(sheet), "hourly rows in sheet 'data'")
add("sensor_variables", len(cfg.sensors), "process variables used")
add("sensor_start", raw.index.min())
add("sensor_end", raw.index.max())
steps = raw.index.to_series().diff().dropna().dt.total_seconds() / 3600
add("sensor_step_hours_most_common", steps.mode().iloc[0])
add("sensor_gaps_over_1h", int((steps > 1).sum()))
add("si_rows_raw", len(si_raw), "rows in sheet 'Si'")
add("si_rows_valid_positive", len(si), "after removing missing and Si <= 0")
add("si_removed_nonpositive_or_missing", len(si_raw) - len(si))
gaps = si.dt.diff().dropna().dt.total_seconds() / 3600
for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
    add(f"si_gap_hours_q{int(q*100)}", round(gaps.quantile(q), 2))
add("si_mean", round(si.Si.mean(), 4)); add("si_std", round(si.Si.std(), 4))
add("si_min", si.Si.min()); add("si_max", si.Si.max())
add("si_above_1.2_total", int((si.Si > 1.2).sum()))
add("samples_after_warmup_and_history", n, "common row universe for all experiments")
add("dev_samples", n_dev); add("dev_start", dev.dt.min()); add("dev_end", dev.dt.max())
add("test_samples", len(test)); add("test_start", test.dt.min()); add("test_end", test.dt.max())
add("test_samples_si_above_1.2", int((test.Si > 1.2).sum()))
add("dev_samples_si_above_1.2", int((dev.Si > 1.2).sum()))
sid = si.copy(); sid["prev"] = sid.Si.shift(1)
on = sid[(sid.Si > 1.2) & (sid.prev <= 1.0)]
add("upset_onsets_total", len(on), "first result above 1.2 right after a result <= 1.0")
add("upset_onsets_dev", int((on.dt <= dev.dt.max()).sum()))
add("upset_onsets_test", int((on.dt > dev.dt.max()).sum()))
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/data_facts.csv", index=False)

# cleaner report: fitted on the development period only (as in the final stage of the experiments)
cl = SensorCleaner().fit(raw.loc[:dev.dt.max()], cfg.sensors)
rep = cl.report().reset_index().rename(columns={"index": "variable"})
rep["rows_in_fit"] = len(raw.loc[:dev.dt.max()])
rep.to_csv(f"{OUT_DIR}/cleaner_report.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("\nCleaner report (fit on development period):\n", rep.to_string(index=False))
