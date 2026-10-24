"""
make_furnace_example.py - writes an example "Furnace B" file pair for trying the "Train for another furnace" tab.

  python make_furnace_example.py        -> data/furnace_b_sensors.csv and data/furnace_b_si.csv

IMPORTANT: this is NOT a second furnace. It is our own dataset (the last two years), reformatted the way another
plant might export it: different column names with units, fewer sensors (no top-gas temperatures, four peripheral
thermocouples dropped), day-first timestamps, and the Si results in a separate file with their own column names.
It shows that the mapping and training work with someone else's file layout; it says nothing about how well the
method transfers to a different furnace. The files go to data/, which is not committed (the dataset is not ours
to redistribute).
"""
import os

import pandas as pd

from run_experiments import DATA_FILE

RENAME = {
    "Fb": "Blast flow (Nm3/min)", "Ph": "Hot blast pressure (bar)", "Pc": "Cold blast pressure (bar)",
    "Tc": "Cold blast temp (C)", "Fo": "Oxygen flow (Nm3/h)", "dP": "Total pressure drop (bar)",
    "dPu": "Upper pressure drop (bar)", "dPl": "Lower pressure drop (bar)", "Pt": "Top pressure (bar)",
    "Th": "Hot blast temp (C)", "CO2": "Top gas CO2 (%)", "H2": "Top gas H2 (%)",
    "Tp1": "Shaft TC 1 (C)", "Tp2": "Shaft TC 2 (C)", "Tp3": "Shaft TC 3 (C)", "Tp4": "Shaft TC 4 (C)",
    "Tp5": "Shaft TC 5 (C)", "Tp6": "Shaft TC 6 (C)", "R": "Ore/coke ratio",
}
START = "2014-05-18"


def main():
    sensor = pd.read_excel(DATA_FILE, sheet_name="data")
    si = pd.read_excel(DATA_FILE, sheet_name="Si")
    sensor["dt"], si["dt"] = pd.to_datetime(sensor["dt"]), pd.to_datetime(si["dt"])
    sensor = sensor[sensor.dt >= START]
    si = si[si.dt >= START]
    out = sensor[["dt", *RENAME]].rename(columns=RENAME)
    out.insert(0, "Timestamp", out.pop("dt").dt.strftime("%d/%m/%Y %H:%M"))
    q = si[["dt", "Si"]].rename(columns={"Si": "Hot metal Si [%]"})
    q.insert(0, "Sample time", q.pop("dt").dt.strftime("%d/%m/%Y %H:%M"))
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/furnace_b_sensors.csv", index=False)
    q.to_csv("data/furnace_b_si.csv", index=False)
    print(f"wrote data/furnace_b_sensors.csv ({len(out):,} rows, {len(RENAME)} sensors) and data/furnace_b_si.csv "
          f"({len(q):,} results), {START} to {sensor.dt.max():%Y-%m-%d}")
    print("Reminder: this is our own furnace reformatted, for trying the training tab; it is not a second furnace.")


if __name__ == "__main__":
    main()
