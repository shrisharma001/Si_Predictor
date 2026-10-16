"""
test_api.py - tests the prediction API without starting a server.

  python make_sample_request.py     # once, creates examples/sample_request.json
  python train_final.py             # once, creates model/si_pipeline_hybrid.pkl
  python test_api.py                # exits with status 1 if a check fails (the demo checks need data/ and model/si_pipeline_demo.pkl)
"""
import copy
import json
import sys

import numpy as np
import pandas as pd

from app import create_app
try:
    from si_pipeline import require_version
except ImportError:
    raise SystemExit("Your si_pipeline.py is out of date: copy the latest si_pipeline.py from the project folder over yours.")
require_version("1.3")
from si_pipeline import SiPipeline

FAILED = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not cond:
        FAILED.append(name)


app = create_app()
client = app.test_client()
EX = json.load(open("examples/sample_request.json"))


def post(body, **kw):
    return client.post("/api/predict", data=json.dumps(body), content_type="application/json", **kw)


# ---------------------------------------------------------------- basic endpoints
print("\n[1] endpoints")
r = client.get("/health")
check("GET /health returns 200 and says the model is loaded", r.status_code == 200 and r.get_json()["model_loaded"] is True)
r = client.get("/api/model")
check("GET /api/model returns the model card", r.status_code == 200 and "features" in r.get_json())
check("GET /api/example returns the example request", client.get("/api/example").status_code == 200)
check("GET /console serves the developer console", client.get("/console").status_code == 200 and b"Developer console" in client.get("/console").data)
r = client.get("/nope")
check("unknown endpoint gives a JSON 404", r.status_code == 404 and r.get_json()["error"]["code"] == "not_found")
r = client.get("/api/predict")
check("GET on /api/predict gives a JSON 405", r.status_code == 405 and r.get_json()["error"]["code"] == "method_not_allowed")

# ---------------------------------------------------------------- happy path
print("\n[2] valid request")
r = post(EX)
b = r.get_json()
res = b["results"][0]
check("valid request returns 200", r.status_code == 200, b.get("status"))
check("prediction is a plausible Si value (0.05-1.5)", res["prediction"] is not None and 0.05 < res["prediction"] < 1.5, str(res["prediction"]))
check("interval brackets the prediction and has level 0.9", res["lower"] < res["prediction"] < res["upper"] and res["interval_level"] == 0.9,
      f"[{res['lower']}, {res['upper']}]")
check("response carries model metadata and a request id", "trained_through" in b["model"] and len(b["request_id"]) == 8)
# identical to calling the pipeline directly
pipe = SiPipeline.load("model/si_pipeline_hybrid.pkl")
sensor = pd.DataFrame(EX["sensor_readings"]); sensor["dt"] = pd.to_datetime(sensor["dt"])
hist = pd.DataFrame(EX["si_history"]); hist["dt"] = pd.to_datetime(hist["dt"])
direct = pipe.predict_frame(sensor, [pd.Timestamp(EX["query_time"])], si_history=hist).iloc[0]
check("API result equals the direct pipeline call", abs(direct.prediction - res["prediction"]) < 1e-3)
# default query time = latest sensor reading
ex2 = {k: v for k, v in EX.items() if k != "query_time"}
r2 = post(ex2)
check("query time defaults to the latest sensor reading", r2.status_code == 200 and r2.get_json()["results"][0]["time"][:10] == sensor.dt.max().isoformat()[:10])
# several queries in one request
ex3 = copy.deepcopy(EX)
qt = pd.Timestamp(EX["query_time"])
ex3["query_times"] = [(qt - pd.Timedelta(hours=h)).isoformat() for h in (0, 1, 2)]
b3 = post(ex3).get_json()
check("several query times give several results", len(b3["results"]) == 3 and all(x["prediction"] is not None for x in b3["results"]))

# ---------------------------------------------------------------- no leakage from the future
print("\n[3] no information from the future")
fut = copy.deepcopy(EX)
fut["si_history"].append({"dt": (qt + pd.Timedelta(hours=1)).isoformat(), "Si": 9.9})
fut["si_history"].append({"dt": qt.isoformat(), "Si": 9.9})              # a result stamped exactly at the query time
pf = post(fut).get_json()["results"][0]["prediction"]
check("future (and same-time) Si results do not change the prediction", abs(pf - res["prediction"]) < 1e-9, f"{pf} vs {res['prediction']}")

# ---------------------------------------------------------------- validation errors
print("\n[4] validation")
r = client.post("/api/predict", data="not json", content_type="text/plain")
check("wrong content type gives 415", r.status_code == 415 and r.get_json()["error"]["code"] == "unsupported_media_type")
r = client.post("/api/predict", data="{broken", content_type="application/json")
check("malformed JSON gives 400", r.status_code == 400 and r.get_json()["error"]["code"] == "invalid_json")
bad = copy.deepcopy(EX); [d.pop("Tp4") for d in bad["sensor_readings"]]
r = post(bad)
check("missing sensor field gives 400 naming the field", r.status_code == 400 and "Tp4" in r.get_json()["error"]["details"]["missing"])
bad = {k: v for k, v in EX.items() if k != "si_history"}
check("missing si_history gives 400", post(bad).status_code == 400)
bad = copy.deepcopy(EX); bad["sensor_readings"][3]["dt"] = "not a date"
r = post(bad)
check("unreadable timestamp gives 400", r.status_code == 400 and r.get_json()["error"]["code"] == "invalid_timestamp")
bad = copy.deepcopy(EX); bad["sensor_readings"] = []
check("empty sensor_readings gives 400", post(bad).status_code == 400)
bad = copy.deepcopy(EX); bad["query_time"] = "yesterday-ish"
check("unreadable query time gives 400", post(bad).status_code == 400)
small = create_app(); small.config["MAX_CONTENT_LENGTH"] = 2000
r = small.test_client().post("/api/predict", data=json.dumps(EX), content_type="application/json")
check("oversized body gives 413", r.status_code == 413 and r.get_json()["error"]["code"] == "payload_too_large")

# ---------------------------------------------------------------- data-quality handling
print("\n[5] data quality")
short = copy.deepcopy(EX); short["sensor_readings"] = short["sensor_readings"][-5:]
r = post(short)
check("too little sensor history gives 422 with a reason", r.status_code == 422 and r.get_json()["status"] == "no_prediction"
      and "history" in r.get_json()["results"][0]["messages"][0], r.get_json()["results"][0]["messages"][0])
few = copy.deepcopy(EX); few["si_history"] = few["si_history"][:2]
r = post(few)
check("fewer than 3 previous Si results gives 422", r.status_code == 422, "; ".join(r.get_json()["results"][0]["messages"]))
fault = copy.deepcopy(EX); fault["sensor_readings"][-1]["Tp4"] = 3311.2
r = post(fault)
m = r.get_json()["results"][0]
check("a fault code is flagged and the prediction stays close", r.status_code == 200 and any("Tp4" in x for x in m["messages"])
      and abs(m["prediction"] - res["prediction"]) < 0.1, f"{m['prediction']} vs {res['prediction']}")
txt = copy.deepcopy(EX); txt["sensor_readings"][-1]["Fb"] = "abc"
r = post(txt)
check("a non-numeric reading is treated as faulty, not as a crash", r.status_code == 200 and any("Fb" in x for x in r.get_json()["results"][0]["messages"]))
nul = copy.deepcopy(EX); nul["sensor_readings"][-1]["H2"] = None
check("a null reading is handled", post(nul).status_code == 200)
stale = copy.deepcopy(EX); stale["si_history"] = [h for h in stale["si_history"] if pd.Timestamp(h["dt"]) < qt - pd.Timedelta(hours=9)]
r = post(stale)
check("a stale previous Si result gives a warning", r.status_code == 200 and any("old" in x for x in r.get_json()["results"][0]["messages"]) and r.get_json()["status"] == "warning")
extra = copy.deepcopy(EX); extra["unexpected_key"] = {"a": 1}
check("unknown keys in the body are ignored", post(extra).status_code == 200)
unsorted = copy.deepcopy(EX); unsorted["sensor_readings"] = list(reversed(unsorted["sensor_readings"]))
pu = post(unsorted).get_json()["results"][0]["prediction"]
check("sensor rows in any order give the same prediction", abs(pu - res["prediction"]) < 1e-9)
body = json.dumps(post(fault).get_json())
check("responses never contain NaN (valid JSON)", "NaN" not in body and "Infinity" not in body)

# ---------------------------------------------------------------- interface files
print("\n[6] interface")
import os
r = client.get("/")
ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
missing_ui = [f for f in ("index.html", "app.js", "app.css") if not os.path.exists(os.path.join(ui_dir, f))]
check("GET / serves the operator interface", r.status_code == 200 and b"Hot-metal silicon" in r.data,
      f"HTTP {r.status_code}; missing from {ui_dir}: {', '.join(missing_ui)}" if missing_ui else f"HTTP {r.status_code}")
check("GET /console still serves the developer console", client.get("/console").status_code == 200)
check("static script and stylesheet are served", client.get("/static/app.js").status_code == 200 and client.get("/static/app.css").status_code == 200)
check("/api/model lists the required sensor columns", len(client.get("/api/model").get_json()["required_sensor_columns"]) == 27)

# ---------------------------------------------------------------- file upload
print("\n[7] file upload")
import io
sens_df = pd.DataFrame(EX["sensor_readings"]); hist_df = pd.DataFrame(EX["si_history"])


def upload(sensor_bytes, si_bytes, sensor_name="sensors.csv", si_name="si.csv", qt=EX["query_time"]):
    data = {"query_time": qt}
    if sensor_bytes is not None:
        data["sensor_file"] = (io.BytesIO(sensor_bytes), sensor_name)
    if si_bytes is not None:
        data["si_file"] = (io.BytesIO(si_bytes), si_name)
    return client.post("/api/predict/upload", data=data, content_type="multipart/form-data")


csv_s, csv_h = sens_df.to_csv(index=False).encode(), hist_df.to_csv(index=False).encode()
r = upload(csv_s, csv_h)
b = r.get_json()
check("CSV upload returns the same prediction as the JSON endpoint", r.status_code == 200 and abs(b["results"][0]["prediction"] - res["prediction"]) < 1e-9,
      f"{b['results'][0]['prediction']} vs {res['prediction']}")
check("upload response carries the lab-result context for the chart", len(b["context"]["si_history"]) == len(hist_df) and b["context"]["sensor_rows_used"] == len(sens_df))
xb = io.BytesIO()
with pd.ExcelWriter(xb) as w:
    sens_df.to_excel(w, sheet_name="data", index=False)
    hist_df.to_excel(w, sheet_name="Si", index=False)
xbytes = xb.getvalue()
r = upload(xbytes, xbytes, "plant.xlsx", "plant.xlsx")
check("one Excel workbook with both sheets works for both fields", r.status_code == 200 and abs(r.get_json()["results"][0]["prediction"] - res["prediction"]) < 1e-9)
fut = sens_df.copy(); fut["dt"] = pd.to_datetime(fut["dt"]) + pd.Timedelta(hours=72); fut[pipe.cfg.sensors] = 99999.0
mix = sens_df.copy(); mix["dt"] = pd.to_datetime(mix["dt"]).dt.strftime("%Y-%m-%d %H:%M:%S"); mix.loc[::2, "dt"] = sens_df.loc[::2, "dt"]
r = upload(mix.to_csv(index=False).encode(), csv_h)
check("a file mixing timestamp styles is read correctly", r.status_code == 200 and abs(r.get_json()["results"][0]["prediction"] - res["prediction"]) < 1e-9)
r = upload(pd.concat([sens_df, fut.astype({"dt": str})]).to_csv(index=False).encode(), csv_h)
check("sensor rows after the chosen time are ignored (no leakage)", r.status_code == 200 and abs(r.get_json()["results"][0]["prediction"] - res["prediction"]) < 1e-9)
r = upload(csv_s, None)
check("a missing file gives 400 naming it", r.status_code == 400 and r.get_json()["error"]["code"] == "missing_file" and "Si" in r.get_json()["error"]["message"])
r = upload(b"%PDF-1.4", csv_h, sensor_name="readings.pdf")
check("an unsupported file type gives 400", r.status_code == 400 and r.get_json()["error"]["code"] == "unsupported_file_type")
r = upload(sens_df.drop(columns=["Tp4"]).to_csv(index=False).encode(), csv_h)
check("a sensor file missing a column gives 400 naming it", r.status_code == 400 and "Tp4" in r.get_json()["error"]["details"]["missing"])
r = upload(b"dt,Fb\n", csv_h)
check("an empty or header-only file gives 400", r.status_code == 400)
r = upload(b"\x00\x01garbage", csv_h, sensor_name="broken.xlsx")
check("an unreadable workbook gives 400, not a crash", r.status_code == 400 and r.get_json()["error"]["code"] == "unreadable_file")
r = upload(csv_s, csv_h, qt="2010-01-01T00:00")
check("a time with no sensor data before it gives 422", r.status_code == 422)

# ---------------------------------------------------------------- held-out demo replay
print("\n[8] demo replay")
info = client.get("/api/demo/info").get_json()
check("demo replay is available", info["available"] is True, info.get("reason", ""))
if info["available"]:
    r = client.post("/api/demo/replay", json={"start": "2016-03-10T00:00", "casts": 48})
    b = r.get_json()
    check("a 48-cast replay returns 48 rows with measured values", r.status_code == 200 and len(b["results"]) == 48 and all(x["actual"] is not None for x in b["results"]))
    check("the replay beats persistence in normal operation", b["summary"]["mae"] < b["summary"]["persistence_mae"],
          f"{b['summary']['mae']} vs {b['summary']['persistence_mae']}")
    check("replayed casts all lie after the demo model's training data", min(pd.Timestamp(x["time"]) for x in b["results"]) > pd.Timestamp(b["model"]["trained_through"]))
    one = client.post("/api/demo/replay", json={"start": b["results"][10]["time"], "casts": 1}).get_json()["results"][0]
    check("a cast predicted alone equals the same cast inside a longer replay (no leakage)",
          one["time"] == b["results"][10]["time"] and abs(one["prediction"] - b["results"][10]["prediction"]) < 1e-9)
    check("a start outside the held-out period gives 400", client.post("/api/demo/replay", json={"start": "2014-05-01", "casts": 5}).status_code == 400)
    check("0 or 999 casts give 400", client.post("/api/demo/replay", json={"start": "2016-03-10", "casts": 0}).status_code == 400
          and client.post("/api/demo/replay", json={"start": "2016-03-10", "casts": 999}).status_code == 400)
    check("an unreadable start gives 400", client.post("/api/demo/replay", json={"start": "soon", "casts": 5}).status_code == 400)
nodemo = create_app(demo_model_path="model/does_not_exist.pkl").test_client()
i2 = nodemo.get("/api/demo/info").get_json()
check("without a demo model, info explains how to create it", i2["available"] is False and "train_final.py" in i2["reason"])
check("without a demo model, replay gives 503", nodemo.post("/api/demo/replay", json={"start": "2016-03-10", "casts": 5}).status_code == 503)

print("\nRESULT:", "ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} CHECK(S) FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)