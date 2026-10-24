"""
test_furnace.py - checks for training on another furnace (furnace.py and the /api/furnaces routes).

  python test_furnace.py

Needs no dataset: it builds a synthetic "Furnace S" with its own column names, 10-minute readings at irregular
times, gaps, a fault code and day-first timestamps, so it exercises the parts that differ from our furnace
(column mapping, date parsing, hourly resampling). The synthetic numbers check the plumbing and the leakage rules,
not accuracy. The built-in model and its demo are not touched.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import furnace as F

FAILED = []


def check(ok, label, detail=""):
    print(("PASS  " if ok else "FAIL  ") + label + (f"  [{detail}]" if detail != "" else ""))
    if not ok:
        FAILED.append(label)


def synthetic_furnace(days=330, seed=7):
    """Si follows a slow 'heat' state that the sensors partly see, plus noise; lab results every 1.2-2.4 h."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-01-01", periods=days * 144, freq="10min") + pd.to_timedelta(rng.integers(0, 50, days * 144), unit="s")
    n = len(t)
    heat = np.zeros(n)
    for i in range(1, n):
        heat[i] = 0.9995 * heat[i - 1] + rng.normal(0, 0.02)
    s = pd.DataFrame({
        "Zeitstempel": t.strftime("%d.%m.%Y %H:%M:%S"),
        "Wind volume [m3/min]": 4000 + 150 * heat + rng.normal(0, 40, n),
        "Hot blast T (degC)": 1150 + 30 * heat + rng.normal(0, 8, n),
        "Top gas CO": 23 - 0.8 * heat + rng.normal(0, 0.3, n),
        "Burden probe 1": 300 + rng.normal(0, 25, n),
        "Spare channel": rng.normal(0, 1, n),
    })
    s.loc[rng.choice(n, 400, replace=False), "Top gas CO"] = -99.0           # a fault code
    s.loc[(t >= "2021-06-01") & (t < "2021-06-01 07:00"), "Wind volume [m3/min]"] = 0.0   # blast off
    s = s[~((t >= "2021-08-10") & (t < "2021-08-11 12:00"))]                # a 36 h logging gap
    q_times = [pd.Timestamp("2021-01-01 03:00")]
    while q_times[-1] < t[-1] - pd.Timedelta(hours=3):
        q_times.append(q_times[-1] + pd.Timedelta(minutes=int(rng.integers(72, 145))))
    idx = np.searchsorted(t.values, pd.DatetimeIndex(q_times).values) - 1
    si = 0.5 + 0.18 * heat[np.clip(idx - 18, 0, None)] + rng.normal(0, 0.05, len(idx))
    si = np.clip(si, 0.05, None)
    q = pd.DataFrame({"Probe time": pd.DatetimeIndex(q_times).strftime("%d/%m/%Y %H:%M"), "Silicon wt%": np.round(si, 3)})
    return s, q


def main():
    root = tempfile.mkdtemp(prefix="si_furnace_test_")
    try:
        run(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\nRESULT:", "ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


def run(root):
    print("[1] dates and names")
    s = pd.Series(["01/02/2021 00:00", "13/02/2021 01:00"])
    check(F.parse_time(s).iloc[0] == pd.Timestamp("2021-02-01"), "day-first dates found from later rows (13/02)")
    s = pd.Series(["02/01/2021 00:00", "02/13/2021 01:00"])
    check(F.parse_time(s, dayfirst=True).iloc[0] == pd.Timestamp("2021-02-01"), "month-first dates found from later rows (02/13)")
    s = pd.Series(["01/02/2021 00:00", "03/04/2021 01:00"])
    check(F.parse_time(s, True).iloc[0] == pd.Timestamp("2021-02-01") and F.parse_time(s, False).iloc[0] == pd.Timestamp("2021-01-02"),
          "ambiguous dates follow the day-first setting")
    check(F.parse_time(pd.Series(["2021-03-04T05:06:00+01:00"])).iloc[0] == pd.Timestamp("2021-03-04 04:06"),
          "time zones are converted to UTC")
    taken = set()
    names = [F.safe_name(x, taken) for x in ["Wind volume [m3/min]", "Wind volume (m3/min)", "1st probe", "Si"]]
    check(names == ["Wind_volume_m3_min", "Wind_volume_m3_min_2", "s_1st_probe", "Si_2"], "safe internal names are unique", names)

    print("\n[2] inspect a synthetic furnace (10-minute irregular readings, own column names)")
    sensor, si = synthetic_furnace()
    s_csv, q_xlsx = io.BytesIO(), io.BytesIO()
    sensor.to_csv(s_csv, index=False)
    with pd.ExcelWriter(q_xlsx) as w:
        pd.DataFrame({"note": ["lab export"]}).to_excel(w, sheet_name="About", index=False)
        si.to_excel(w, sheet_name="Lab", index=False)
    t0 = time.time()
    ins = F.inspect_upload(root, s_csv.getvalue(), "furnace_s.csv", q_xlsx.getvalue(), "lab.xlsx")
    sug = ins["suggested"]
    check(sug["sensor"]["time_col"] == "Zeitstempel" and sug["si"]["time_col"] == "Probe time", "time columns found", (sug["sensor"]["time_col"], sug["si"]["time_col"]))
    check(sug["si"]["sheet"] == "Lab" and sug["si"]["value_col"] == "Silicon wt%", "Si sheet and column found", (sug["si"]["sheet"], sug["si"]["value_col"]))
    check(sug["sensor"]["dayfirst"] and sug["si"]["dayfirst"], "day-first suggested for both files")
    check(sug["sensor"]["blast_off_col"] == "Wind volume [m3/min]", "blast-flow column guessed", sug["sensor"]["blast_off_col"])
    check(len(sug["sensor"]["columns"]) == 5, "all numeric sensors suggested", len(sug["sensor"]["columns"]))
    print(f"      inspect took {time.time() - t0:.1f} s")

    print("\n[3] mapping errors are explained")
    def err(fn):
        try:
            fn()
        except F.FurnaceError as e:
            return e.code
        return None
    bad = json.loads(json.dumps(sug)); bad["sensor"]["blast_off_col"] = "Burden probe 1"; bad["sensor"]["columns"] = ["Top gas CO"]
    check(err(lambda: F.start_job(root, ins["upload_id"], "S", bad)) == "invalid_mapping", "blast column must be a selected sensor")
    bad = json.loads(json.dumps(sug)); bad["si"]["value_col"] = "nope"
    check(err(lambda: F.start_job(root, ins["upload_id"], "S", bad)) == "missing_columns", "unknown column is reported")
    check(err(lambda: F.start_job(root, "f" * 32, "S", sug)) == "upload_expired", "unknown upload is reported")
    check(err(lambda: F.start_job(root, ins["upload_id"], "", sug)) == "invalid_name", "a name is required")
    check(err(lambda: F.furnace_dir(root, "../etc")) == "invalid_furnace", "furnace ids cannot leave the folder")

    print("\n[4] train in the background and evaluate in time order")
    fid, ctx = F.start_job(root, ins["upload_id"], "Furnace S", sug)
    check(err(lambda: F.start_job(root, ins["upload_id"], "Furnace S2", sug)) == "busy", "a second job waits for the first")
    t0 = time.time()
    F.run_training_job(root, fid, ctx)
    job = F.job_status(root, fid)
    check(job["status"] == "done", "training job finished", job.get("error"))
    print(f"      training took {time.time() - t0:.1f} s")
    if job["status"] != "done":
        return
    pipe, card = F.load_furnace(root, fid)
    ev = card["evaluation"]
    times = [(f["from"], f["to"]) for f in ev["folds"]] + [(ev["test"]["from"], ev["test"]["to"])]
    check(all(pd.Timestamp(times[i][1]) < pd.Timestamp(times[i + 1][0]) for i in range(len(times) - 1)),
          "folds and the held-out period follow each other in time")
    check(ev["test"]["model"]["mae"] < ev["test"]["persistence"]["mae"], "on this synthetic furnace the model beats persistence",
          f'{ev["test"]["model"]["mae"]:.4f} vs {ev["test"]["persistence"]["mae"]:.4f}')
    check(0.8 <= ev["test"]["coverage"] <= 0.97, "held-out interval coverage near 90 %", round(ev["test"]["coverage"], 3))
    check(any(f["sensor"] == "Top gas CO" for f in card["sensor_faults"]), "the fault code was flagged under the furnace's own name",
          card["sensor_faults"][:2])
    check(card["required_columns"]["sensor"][0] == "Zeitstempel", "card lists the furnace's own column names")
    check(pipe.cfg.resample_hourly and pipe.cfg.blast_off_col == "Wind_volume_m3_min", "pipeline resamples and knows the blast column")
    check(not os.path.exists(os.path.join(root, "furnaces", ".training.lock")), "training lock released")
    listed = F.list_furnaces(root)
    check(len(listed) == 1 and listed[0]["verdict"] in ("good", "weak", "poor"), "furnace listed with a verdict")

    print("\n[5] predictions use only the past")
    s_map, q_map, _ = F.apply_mapping(sensor, si, card["mapping"])
    qt = pd.Timestamp("2021-10-20 10:30")
    base = pipe.predict_frame(s_map[s_map.dt <= qt], [qt], si_history=q_map[q_map.dt < qt])
    later = pipe.predict_frame(s_map[s_map.dt <= qt + pd.Timedelta(hours=6)], [qt], si_history=q_map)
    check(base.prediction.notna().all() and np.allclose(base.prediction, later.prediction),
          "readings and lab results after the query time do not change the prediction",
          f"{base.prediction.iloc[0]:.4f} vs {later.prediction.iloc[0]:.4f}")
    s_fault = s_map.copy()
    hour_end = qt.floor("h")                       # the hourly bin the prediction uses: (09:00, 10:00] for 10:30
    dead = (s_fault.dt > hour_end - pd.Timedelta(hours=1)) & (s_fault.dt <= hour_end)
    s_fault.loc[dead, "Top_gas_CO"] = -99.0        # the sensor sends its fault code for the whole hour
    out = pipe.predict_frame(s_fault[s_fault.dt <= qt], [qt], si_history=q_map[q_map.dt < qt])
    one = s_map.copy()
    one.loc[one.index[dead.to_numpy()][0], "Top_gas_CO"] = -99.0      # a single bad 10-minute reading
    out1 = pipe.predict_frame(one[one.dt <= qt], [qt], si_history=q_map[q_map.dt < qt])
    check(abs(out1.prediction.iloc[0] - base.prediction.iloc[0]) < 1e-9, "one bad 10-minute reading is absorbed by the hourly median")
    msg = F.translate_messages(out.messages.iloc[0].split("; ")[0], card) if out.messages.iloc[0] else ""
    check("Top gas CO" in msg, "a fault code in the latest hour is reported under the furnace's own name", msg)

    print("\n[6] data the protocol cannot evaluate is refused")
    small_s, small_q = synthetic_furnace(days=60, seed=3)
    b1, b2 = io.BytesIO(), io.BytesIO()
    small_s.to_csv(b1, index=False); small_q.to_csv(b2, index=False)
    ins2 = F.inspect_upload(root, b1.getvalue(), "s.csv", b2.getvalue(), "q.csv")
    fid2, ctx2 = F.start_job(root, ins2["upload_id"], "Too short", ins2["suggested"])
    F.run_training_job(root, fid2, ctx2)
    j2 = F.job_status(root, fid2)
    check(j2["status"] == "failed" and j2["error"]["code"] == "too_few_samples", "too few Si results", j2.get("error", {}).get("message", "")[:70])
    sensor_big, si_big = synthetic_furnace()
    hb = sensor_big.copy()
    hb.loc[hb.index % 18 != 0, "Burden probe 1"] = np.nan                     # the probe reports only every third hour
    b1, b2 = io.BytesIO(), io.BytesIO()
    hb.to_csv(b1, index=False); si_big.to_csv(b2, index=False)
    ins3 = F.inspect_upload(root, b1.getvalue(), "s.csv", b2.getvalue(), "q.csv")
    m3 = ins3["suggested"]; m3["sensor"]["columns"] = list(dict.fromkeys(m3["sensor"]["columns"] + ["Burden probe 1"]))
    fid3, ctx3 = F.start_job(root, ins3["upload_id"], "Holes", m3)
    F.run_training_job(root, fid3, ctx3)
    j3 = F.job_status(root, fid3)
    check(j3["status"] == "failed" and j3["error"]["code"] == "sensors_unusable" and j3["error"]["details"]["sensors"] == ["Burden probe 1"],
          "a mostly empty sensor is named so it can be deselected", j3.get("error", {}).get("details"))

    print("\n[7] HTTP routes (Flask test client)")
    try:
        from app import create_app
    except SystemExit as e:
        print("      skipped: the built-in model is needed to start the app:", e)
        return
    app = create_app(furnace_dir=root)
    c = app.test_client()
    r = c.get("/api/furnaces")
    check(r.status_code == 200 and any(f["furnace_id"] == fid for f in r.get_json()["furnaces"]), "GET /api/furnaces lists the furnace")
    r = c.get(f"/api/furnaces/{fid}")
    body = r.get_json()
    check(r.status_code == 200 and len(body["test_series"]["time"]) == ev["test"]["n"], "GET /api/furnaces/<id> returns the report and series")
    data = {"sensor_file": (io.BytesIO(s_csv.getvalue()), "s.csv"), "si_file": (io.BytesIO(q_xlsx.getvalue()), "lab.xlsx"),
            "query_time": "2021-10-20T10:30"}
    r = c.post(f"/api/furnaces/{fid}/predict/upload", data=data, content_type="multipart/form-data")
    b = r.get_json()
    check(r.status_code == 200 and b["results"][0]["prediction"] is not None, "predict with the furnace's own files", b.get("status"))
    check(abs(b["results"][0]["prediction"] - base.prediction.iloc[0]) < 1e-3, "the upload route gives the same prediction as the pipeline")
    data = {"sensor_file": (io.BytesIO(b"dt,Fb\n2021-01-01 00:00,1\n"), "s.csv"), "si_file": (io.BytesIO(q_xlsx.getvalue()), "lab.xlsx")}
    r = c.post(f"/api/furnaces/{fid}/predict/upload", data=data, content_type="multipart/form-data")
    check(r.status_code == 400 and r.get_json()["error"]["code"] == "missing_columns", "files in the wrong layout are refused")
    check(c.get("/api/furnaces/nope/job").status_code == 404, "unknown job gives 404")
    r = c.post("/api/furnaces/inspect", data={"sensor_file": (io.BytesIO(b"x"), "a.pdf"), "si_file": (io.BytesIO(b"x"), "b.csv")},
               content_type="multipart/form-data")
    check(r.status_code == 400 and r.get_json()["error"]["code"] == "unsupported_file_type", "unsupported file type refused")
    r = c.delete(f"/api/furnaces/{fid}")
    check(r.status_code == 200 and not os.path.exists(os.path.join(root, "furnaces", fid)), "DELETE removes the furnace")


if __name__ == "__main__":
    main()
