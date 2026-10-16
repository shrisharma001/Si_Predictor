"""predict_example.py - how the saved pipeline is used (this is what the API/UI will call).

  python predict_example.py
"""
import pandas as pd

from run_experiments import DATA_FILE
try:
    from si_pipeline import require_version
except ImportError:
    raise SystemExit("Your si_pipeline.py is out of date: copy the latest si_pipeline.py from the project folder over yours.")
require_version("1.3")
from si_pipeline import SiPipeline

pipe = SiPipeline.load("model/si_pipeline_hybrid.pkl")
sensor = pd.read_excel(DATA_FILE, sheet_name="data")
si = pd.read_excel(DATA_FILE, sheet_name="Si")
sensor["dt"] = pd.to_datetime(sensor["dt"]); si["dt"] = pd.to_datetime(si["dt"])

# NOTE: these three dates were part of the training data of the final model, so this demonstrates the calls,
# not the accuracy. For accuracy see results/consolidated_results.csv.
# Ask for the last three lab results one by one, using only information available before each one:
queries = si.dt.sort_values().iloc[-3:]
for t in queries:
    window = sensor[(sensor.dt <= t) & (sensor.dt > t - pd.Timedelta(hours=72))]      # last 72 hourly readings
    history = si[si.dt < t]                                                           # only PAST lab results
    r = pipe.predict_frame(window, [t], si_history=history).iloc[0]
    actual = si.loc[si.dt == t, "Si"].iloc[0]
    print(f"{t}  predicted {r.prediction:.3f}  {r.interval_level:.0%} interval [{r.lower:.3f}, {r.upper:.3f}]  "
          f"actual {actual:.3f}  status={r.status}  {r.messages}")
