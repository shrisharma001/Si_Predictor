"""
app.py - prediction API and operator interface around the saved SiPipeline (hybrid model + 90 % conformal intervals).

Run (development / local demo):   python app.py            then open http://127.0.0.1:5000
Run (production, Linux/macOS):    gunicorn -w 2 -b 0.0.0.0:8000 app:app

Environment variables
  SI_MODEL_PATH   live model                      (default model/si_pipeline_hybrid.pkl)
  SI_MODEL_CARD   live model card                 (default model/model_card.json)
  SI_DEMO_MODEL   held-out demo model             (default model/si_pipeline_demo.pkl, made by train_final.py)
  SI_DEMO_DATA    dataset used by the demo replay (default data/1_blast_furnace_data_first_dataset.xlsx)
  SI_EXAMPLE      example request                 (default examples/sample_request.json)
  SI_FURNACE_DIR  where furnace models and uploads are kept (default: the folder of SI_MODEL_PATH)
  HOST, PORT      bind address for `python app.py` (default 127.0.0.1, 5000)
  FLASK_DEBUG=1   Flask debug mode (development only; never combine with a public HOST)

Endpoints
  GET  /                     operator interface (static/index.html)
  GET  /console              raw JSON developer console
  GET  /health               liveness + which models are loaded
  GET  /api/model            model card
  GET  /api/example          example request body for /api/predict
  POST /api/predict          JSON: sensor_readings, si_history, optional query_time(s)
  POST /api/predict/upload   multipart: sensor_file and si_file (CSV or Excel), optional query_time
  GET  /api/demo/info        is the held-out replay available, and for which period
  POST /api/demo/replay      JSON: start, casts  ->  held-out predictions next to the measured values

  Other furnaces (furnace.py): train and validate the same recipe on another furnace's own history
  POST   /api/furnaces/inspect             multipart: sensor_file, si_file -> columns and a suggested mapping
  POST   /api/furnaces/train               JSON: upload_id, name, mapping -> 202, runs in the background
  GET    /api/furnaces/<id>/job            progress of a training job
  GET    /api/furnaces                     trained furnace models
  GET    /api/furnaces/<id>                model card with the evaluation report and the test-period series
  DELETE /api/furnaces/<id>                delete a furnace model
  POST   /api/furnaces/<id>/predict/upload multipart: sensor_file, si_file (that furnace's own columns), query_time

  Model health (health.py): drift checks on predictions that were later compared with the lab result
  GET    /api/health/demo                  the demo model over the whole held-out period
  GET    /api/furnaces/<id>/health         a furnace model over its held-out period

Security note: model files are pickles. Only load model files you created yourself.
"""
import io
import json
import logging
import os
import threading
import time
import uuid

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

try:
    from si_pipeline import require_version
except ImportError:
    raise SystemExit("Your si_pipeline.py is out of date: copy the latest si_pipeline.py from the project folder over yours.")
require_version("1.4")
from si_pipeline import SiPipeline, prepare_si_frame
import furnace as fz
import health as hz

MAX_BODY_BYTES = 64 * 1024 * 1024    # a few years of another furnace's history
MAX_SENSOR_ROWS = 5000          # JSON requests
MAX_HISTORY_ROWS = 5000
MAX_QUERIES = 500
MAX_UPLOAD_ROWS = 60000         # uploaded files are trimmed to the window the model needs
UPLOAD_SENSOR_WINDOW = pd.Timedelta(days=7)
CONTEXT_RESULTS = 20            # previous Si results returned for charting
MAX_REPLAY_CASTS = 200
REPLAY_LEAD_IN = 12

log = logging.getLogger("si_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
HERE = os.path.dirname(os.path.abspath(__file__))


class ApiError(Exception):
    def __init__(self, status, code, message, details=None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def num(x, nd=4):
    """JSON-safe number: NaN/inf become null."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def iso(t):
    return pd.Timestamp(t).isoformat()


def create_app(model_path=None, card_path=None, example_path=None, demo_model_path=None, demo_data_path=None,
               preload_demo=False, furnace_dir=None):
    app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    model_path = model_path or os.environ.get("SI_MODEL_PATH", "model/si_pipeline_hybrid.pkl")
    card_path = card_path or os.environ.get("SI_MODEL_CARD", "model/model_card.json")
    example_path = example_path or os.environ.get("SI_EXAMPLE", "examples/sample_request.json")
    demo_model_path = demo_model_path or os.environ.get("SI_DEMO_MODEL", "model/si_pipeline_demo.pkl")
    demo_data_path = demo_data_path or os.environ.get("SI_DEMO_DATA", "data/1_blast_furnace_data_first_dataset.xlsx")
    furnace_root = furnace_dir or os.environ.get("SI_FURNACE_DIR") or (os.path.dirname(model_path) or ".")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}. Create it with `python train_final.py`.")
    pipe = SiPipeline.load(model_path)
    card = json.load(open(card_path)) if os.path.exists(card_path) else {}
    lock = threading.Lock()
    cfg = pipe.cfg
    log.info("model loaded from %s (trained through %s, %s features)", model_path,
             pipe.train_range_[1], len(pipe.feature_names_))
    demo = {"loaded": False, "error": None}
    demo_lock = threading.Lock()

    # ------------------------------------------------------------------ shared prediction step
    def run_predictions(model, sensor, hist, queries, furnace_card=None):
        kw = {}
        if furnace_card:       # thresholds learned from that furnace's own history
            kw = {"stale_hours": furnace_card["stale_hours"], "upset_level": furnace_card["upset_level"]}
        with lock:
            res = model.predict_frame(sensor, queries, si_history=hist, **kw)
        results = []
        for r in res.itertuples():
            msgs = [m for m in r.messages.split("; ") if m]
            if furnace_card:
                msgs = [fz.translate_messages(m, furnace_card) for m in msgs]
            results.append({"time": iso(r.time), "prediction": num(r.prediction), "lower": num(r.lower),
                            "upper": num(r.upper), "interval_level": num(r.interval_level, 2), "status": r.status,
                            "messages": msgs})
        statuses = [x["status"] for x in results]
        if all(s == "no_prediction" for s in statuses):
            return results, "no_prediction", 422
        if any(s == "no_prediction" for s in statuses):
            return results, "partial", 200
        if any(s == "warning" for s in statuses):
            return results, "warning", 200
        return results, "ok", 200

    def model_meta(model, name, model_card=None):
        meta = {"name": name, "trained_through": str(model.train_range_[1]),
                "interval_level": num(1 - model.alpha, 2) if model.q_ is not None else None}
        rng = (model_card or {}).get("typical_si_range")
        if rng:
            meta["typical_si_range"] = rng       # middle 80 % of the Si results the model was trained on
        return meta

    def si_context(hist, before, n=CONTEXT_RESULTS):
        h = prepare_si_frame(hist, cfg)
        h = h[h[cfg.time_col] < before].tail(n)
        return [{"time": iso(t), "si": num(v)} for t, v in zip(h[cfg.time_col], h[cfg.target])]

    # ------------------------------------------------------------------ input parsing
    def frame_from_records(value, name, required, limit):
        if not isinstance(value, list) or len(value) == 0:
            raise ApiError(400, "invalid_" + name, f"'{name}' must be a non-empty list of records")
        if len(value) > limit:
            raise ApiError(400, "too_many_rows", f"'{name}' has {len(value)} records; the limit is {limit}")
        if not all(isinstance(r, dict) for r in value):
            raise ApiError(400, "invalid_" + name, f"every entry of '{name}' must be an object with named fields")
        return check_frame(pd.DataFrame(value), name, required)

    def check_frame(df, name, required):
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ApiError(400, "missing_columns", f"'{name}' is missing required fields", {"missing": missing})
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce", format="mixed")   # files may mix timestamp styles
        bad = int(df["dt"].isna().sum())
        if bad:
            raise ApiError(400, "invalid_timestamp", f"{bad} record(s) in '{name}' have an unreadable 'dt'",
                           {"first_bad_index": int(df["dt"].isna().idxmax())})
        return df

    def parse_json_request():
        if not request.is_json:
            raise ApiError(415, "unsupported_media_type", "Send a JSON body with Content-Type: application/json")
        p = request.get_json(silent=True)
        if not isinstance(p, dict):
            raise ApiError(400, "invalid_json", "The body is not valid JSON or is not a JSON object")
        return p

    def parse_query_times(p, sensor):
        if "query_times" in p:
            qt = p["query_times"]
            if not isinstance(qt, list) or not qt:
                raise ApiError(400, "invalid_query_times", "'query_times' must be a non-empty list of timestamps")
        elif p.get("query_time"):
            qt = [p["query_time"]]
        else:
            qt = [sensor["dt"].max()]          # default: the latest sensor reading
        if len(qt) > MAX_QUERIES:
            raise ApiError(400, "too_many_queries", f"at most {MAX_QUERIES} query times per request")
        q = pd.to_datetime(pd.Series(qt), errors="coerce", format="mixed")
        if q.isna().any():
            raise ApiError(400, "invalid_timestamp", "a query time could not be read")
        return list(q)

    def read_upload(field, label, required):
        f = request.files.get(field)
        if f is None or not f.filename:
            raise ApiError(400, "missing_file", f"Choose a file for the {label}")
        name = f.filename.lower()
        raw = f.read()
        try:
            if name.endswith(".csv") or name.endswith(".txt"):
                df = pd.read_csv(io.BytesIO(raw))
            elif name.endswith(".xlsx") or name.endswith(".xlsm"):
                xl = pd.ExcelFile(io.BytesIO(raw))
                sheet = next((s for s in xl.sheet_names if required[1] in xl.parse(s, nrows=0).columns), xl.sheet_names[0])
                df = xl.parse(sheet)
            else:
                raise ApiError(400, "unsupported_file_type", f"The {label} must be a .csv or .xlsx file")
        except ApiError:
            raise
        except Exception:
            raise ApiError(400, "unreadable_file", f"The {label} could not be read as CSV or Excel")
        if len(df) == 0:
            raise ApiError(400, "empty_file", f"The {label} has no rows")
        if len(df) > MAX_UPLOAD_ROWS:
            raise ApiError(400, "too_many_rows", f"The {label} has {len(df)} rows; the limit is {MAX_UPLOAD_ROWS}")
        return check_frame(df, label, required)

    # ------------------------------------------------------------------ live prediction routes
    @app.route("/api/predict", methods=["POST"])
    def predict():
        rid, t0 = uuid.uuid4().hex[:8], time.time()
        p = parse_json_request()
        sensor = frame_from_records(p.get("sensor_readings"), "sensor_readings", ["dt"] + cfg.sensors, MAX_SENSOR_ROWS)
        if p.get("si_history") is None:
            raise ApiError(400, "missing_si_history", "'si_history' (previous Si lab results with 'dt' and 'Si') is required")
        hist = frame_from_records(p["si_history"], "si_history", ["dt", "Si"], MAX_HISTORY_ROWS)
        queries = parse_query_times(p, sensor)
        results, overall, code = run_predictions(pipe, sensor, hist, queries)
        log.info("request %s: %d sensor rows, %d history rows, %d queries -> %s (%.0f ms)",
                 rid, len(sensor), len(hist), len(queries), overall, (time.time() - t0) * 1000)
        return jsonify({"request_id": rid, "status": overall, "unit": "wt% Si", "results": results,
                        "model": model_meta(pipe, card.get("name", "SiPipeline"), card)}), code

    @app.route("/api/predict/upload", methods=["POST"])
    def predict_upload():
        rid, t0 = uuid.uuid4().hex[:8], time.time()
        sensor = read_upload("sensor_file", "sensor file", ["dt"] + cfg.sensors)
        hist = read_upload("si_file", "Si results file", ["dt", "Si"])
        q = parse_query_times({"query_time": request.form.get("query_time", "")}, sensor)[0]
        sensor = sensor[(sensor["dt"] <= q) & (sensor["dt"] > q - UPLOAD_SENSOR_WINDOW)]
        if len(sensor) == 0:
            raise ApiError(422, "no_sensor_data", "The sensor file has no readings in the 7 days before the chosen time")
        results, overall, code = run_predictions(pipe, sensor, hist, [q])
        log.info("upload %s: %d sensor rows, %d history rows -> %s (%.0f ms)", rid, len(sensor), len(hist), overall,
                 (time.time() - t0) * 1000)
        return jsonify({"request_id": rid, "status": overall, "unit": "wt% Si", "results": results,
                        "context": {"si_history": si_context(hist, q), "sensor_rows_used": int(len(sensor)),
                                    "sensor_from": iso(sensor["dt"].min()), "sensor_to": iso(sensor["dt"].max())},
                        "model": model_meta(pipe, card.get("name", "SiPipeline"), card)}), code

    # ------------------------------------------------------------------ held-out demo replay
    def load_demo():
        with demo_lock:
            if demo["loaded"] or demo["error"]:
                return
            if not os.path.exists(demo_model_path):
                demo["error"] = f"The demo model {demo_model_path} is missing. Create it with `python train_final.py --demo-only`."
                return
            if not os.path.exists(demo_data_path):
                demo["error"] = f"The dataset {demo_data_path} is missing, so there is nothing to replay."
                return
            t0 = time.time()
            dpipe = SiPipeline.load(demo_model_path)
            sensor = pd.read_excel(demo_data_path, sheet_name="data")
            si_raw = pd.read_excel(demo_data_path, sheet_name="Si")
            sensor["dt"] = pd.to_datetime(sensor["dt"])
            si_raw["dt"] = pd.to_datetime(si_raw["dt"])
            si = prepare_si_frame(si_raw, cfg)
            held = si[si[cfg.time_col] > dpipe.train_range_[1]].reset_index(drop=True)
            demo_card_path = os.path.splitext(demo_model_path)[0].replace("si_pipeline_demo", "model_card_demo") + ".json"
            dcard = json.load(open(demo_card_path)) if os.path.exists(demo_card_path) else {}
            demo.update(loaded=True, pipe=dpipe, card=dcard, sensor=sensor.sort_values("dt").reset_index(drop=True), si=si, held=held)
            log.info("demo loaded in %.1f s: %d held-out casts from %s", time.time() - t0, len(held),
                     held[cfg.time_col].min())

    @app.route("/api/demo/info")
    def demo_info():
        load_demo()
        if not demo["loaded"]:
            return jsonify({"available": False, "reason": demo["error"]})
        h = demo["held"][cfg.time_col]
        return jsonify({"available": True, "model": model_meta(demo["pipe"], "held-out demo copy", demo["card"]),
                        "held_out_from": iso(h.min()), "held_out_to": iso(h.max()), "casts": int(len(h)),
                        "max_casts_per_replay": MAX_REPLAY_CASTS, "suggested_starts": [
                            {"label": "Normal operation (March 2016)", "start": "2016-03-10T00:00:00"},
                            {"label": "Largest upset (January 2016)", "start": "2016-01-08T00:00:00"},
                            {"label": "Start of the held-out period", "start": iso(h.min())}]})

    @app.route("/api/demo/replay", methods=["POST"])
    def demo_replay():
        load_demo()
        if not demo["loaded"]:
            raise ApiError(503, "demo_unavailable", demo["error"])
        p = parse_json_request()
        start = pd.to_datetime(p.get("start"), errors="coerce", format="mixed")
        if pd.isna(start):
            raise ApiError(400, "invalid_timestamp", "'start' must be a date and time")
        try:
            n = int(p.get("casts", 24))
        except (TypeError, ValueError):
            raise ApiError(400, "invalid_casts", "'casts' must be a whole number")
        if not 1 <= n <= MAX_REPLAY_CASTS:
            raise ApiError(400, "invalid_casts", f"'casts' must be between 1 and {MAX_REPLAY_CASTS}")
        held, si, sensor, dpipe = demo["held"], demo["si"], demo["sensor"], demo["pipe"]
        lo, hi = held[cfg.time_col].min(), held[cfg.time_col].max()
        if start < lo - pd.Timedelta(days=1) or start > hi:
            raise ApiError(400, "outside_held_out_period",
                           f"Choose a start between {lo:%d %b %Y} and {hi:%d %b %Y}: only that period was held out from the demo model")
        sel = held[held[cfg.time_col] >= start].head(n)
        if len(sel) == 0:
            raise ApiError(400, "no_casts", "There are no casts after that start")
        q = list(sel[cfg.time_col])
        win = sensor[(sensor["dt"] > q[0] - pd.Timedelta(hours=48)) & (sensor["dt"] <= q[-1])]
        hist = si[si[cfg.time_col] < q[-1]].rename(columns={cfg.time_col: "dt", cfg.target: "Si"})
        results, overall, _ = run_predictions(dpipe, win, hist, q)
        all_si = si.set_index(cfg.time_col)[cfg.target]
        prev = all_si.shift(1)
        rows, err, perr, cov = [], [], [], []
        for r in results:
            t = pd.Timestamp(r["time"])
            actual, persist = float(all_si.loc[t]), float(prev.loc[t])
            r.update(actual=num(actual), persistence=num(persist))
            if r["prediction"] is not None:
                err.append(abs(actual - r["prediction"]))
                perr.append(abs(actual - persist))
                cov.append(r["lower"] <= actual <= r["upper"])
            rows.append(r)
        lead = si[si[cfg.time_col] < q[0]].tail(REPLAY_LEAD_IN)
        summary = {"casts": len(rows), "predicted": len(err),
                   "mae": num(np.mean(err)) if err else None, "persistence_mae": num(np.mean(perr)) if perr else None,
                   "coverage": num(np.mean(cov), 3) if cov else None,
                   "upsets": int(sum(1 for r in rows if (r["actual"] or 0) > 1.2))}
        return jsonify({"status": overall, "unit": "wt% Si", "results": rows, "summary": summary,
                        "lead_in": [{"time": iso(t), "si": num(v)} for t, v in zip(lead[cfg.time_col], lead[cfg.target])],
                        "model": model_meta(dpipe, "held-out demo copy", demo["card"])})

    # ------------------------------------------------------------------ model health
    health_cache = {}

    @app.route("/api/health/demo")
    def health_demo():
        load_demo()
        if not demo["loaded"]:
            raise ApiError(503, "demo_unavailable", demo["error"])
        if "demo" not in health_cache:          # the held-out period never changes, so compute it once per worker
            held, si, sensor, dpipe = demo["held"], demo["si"], demo["sensor"], demo["pipe"]
            q = list(held[cfg.time_col])
            win = sensor[(sensor["dt"] > q[0] - pd.Timedelta(hours=48)) & (sensor["dt"] <= q[-1])]
            hist = si[si[cfg.time_col] < q[-1]].rename(columns={cfg.time_col: "dt", cfg.target: "Si"})
            results, _, _ = run_predictions(dpipe, win, hist, q)
            series = si.set_index(cfg.time_col)[cfg.target]
            prev = series.shift(1)
            ok = [r for r in results if r["prediction"] is not None]
            t = pd.DatetimeIndex([r["time"] for r in ok])
            rep_ = hz.health_report(t, series.loc[t], [r["prediction"] for r in ok], [r["lower"] for r in ok],
                                    [r["upper"] for r in ok], prev.loc[t], [hz.faulty_sensors(r["messages"]) for r in ok])
            rep_.update(source={"kind": "demo", "name": "Built-in model, held-out period",
                                "note": "The demo model was trained up to " + str(dpipe.train_range_[1])[:10] +
                                        "; every prediction here uses only data from before its cast."},
                        not_predicted=len(results) - len(ok))
            health_cache["demo"] = rep_
        return jsonify(health_cache["demo"])

    @app.route("/api/furnaces/<fid>/health")
    def furnace_health(fid):
        _, fcard = fz.load_furnace(furnace_root, fid)
        s = fz.read_json(os.path.join(fz.furnace_dir(furnace_root, fid), "test_series.json"))
        if not s:
            raise ApiError(404, "no_series", "This furnace model has no held-out series")
        rep_ = hz.health_report(s["time"], s["actual"], s["prediction"], s["lower"], s["upper"], s["persistence"])
        rep_.update(source={"kind": "furnace", "name": f"{fcard['name']}, held-out period", "furnace_id": fid,
                            "note": "Sensor-fault tracking is not available here: the report keeps the predictions, not the messages."},
                    not_predicted=0)
        return jsonify(rep_)

    if preload_demo:      # read the replay dataset in the background so the first replay does not stall for ~10 s
        threading.Thread(target=load_demo, name="demo-preload", daemon=True).start()

    # ------------------------------------------------------------------ other furnaces
    def upload_bytes(field, label):
        f = request.files.get(field)
        if f is None or not f.filename:
            raise ApiError(400, "missing_file", f"Choose a file for the {label}")
        return f.read(), f.filename

    @app.route("/api/furnaces/inspect", methods=["POST"])
    def furnace_inspect():
        sb, sn = upload_bytes("sensor_file", "sensor history")
        qb, qn = upload_bytes("si_file", "Si results")
        t0 = time.time()
        out = fz.inspect_upload(furnace_root, sb, sn, qb, qn)
        log.info("furnace upload %s inspected in %.1f s (%s, %s)", out["upload_id"][:8], time.time() - t0, sn, qn)
        return jsonify(out)

    @app.route("/api/furnaces/train", methods=["POST"])
    def furnace_train():
        p = parse_json_request()
        fid, ctx = fz.start_job(furnace_root, p.get("upload_id"), p.get("name"), p.get("mapping"))
        threading.Thread(target=fz.run_training_job, args=(furnace_root, fid, ctx), name=f"train-{fid}", daemon=True).start()
        log.info("furnace %s: training started", fid)
        return jsonify({"furnace_id": fid, "status": "queued", "job_url": f"/api/furnaces/{fid}/job"}), 202

    @app.route("/api/furnaces/<fid>/job")
    def furnace_job(fid):
        return jsonify(fz.job_status(furnace_root, fid))

    @app.route("/api/furnaces")
    def furnace_list():
        return jsonify({"furnaces": fz.list_furnaces(furnace_root)})

    @app.route("/api/furnaces/<fid>", methods=["GET"])
    def furnace_card(fid):
        _, fcard = fz.load_furnace(furnace_root, fid)
        series = fz.read_json(os.path.join(fz.furnace_dir(furnace_root, fid), "test_series.json"), {})
        return jsonify(dict(fcard, test_series=series))

    @app.route("/api/furnaces/<fid>", methods=["DELETE"])
    def furnace_delete(fid):
        fz.delete_furnace(furnace_root, fid)
        log.info("furnace %s deleted", fid)
        return jsonify({"deleted": fid})

    def furnace_sheet(sheets, part):
        """The sheet named in the mapping, else the first one that has the time column."""
        if part.get("sheet") in sheets:
            return sheets[part["sheet"]]
        return next((df for df in sheets.values() if part["time_col"] in df.columns), next(iter(sheets.values())))

    @app.route("/api/furnaces/<fid>/predict/upload", methods=["POST"])
    def furnace_predict(fid):
        rid, t0 = uuid.uuid4().hex[:8], time.time()
        fpipe, fcard = fz.load_furnace(furnace_root, fid)
        mp = fcard["mapping"]
        sb, sn = upload_bytes("sensor_file", "sensor file")
        qb, qn = upload_bytes("si_file", "Si results file")
        sensor_raw = furnace_sheet(fz.read_tables(sb, sn), mp["sensor"])
        si_raw = furnace_sheet(fz.read_tables(qb, qn), mp["si"])
        sensor, hist, _ = fz.apply_mapping(sensor_raw, si_raw, mp)
        q = parse_query_times({"query_time": request.form.get("query_time", "")}, sensor)[0]
        sensor = sensor[(sensor["dt"] <= q) & (sensor["dt"] > q - UPLOAD_SENSOR_WINDOW)]
        if len(sensor) == 0:
            raise ApiError(422, "no_sensor_data", "The sensor file has no readings in the 7 days before the chosen time")
        results, overall, code = run_predictions(fpipe, sensor, hist, [q], furnace_card=fcard)
        log.info("furnace %s upload %s: %d sensor rows -> %s (%.0f ms)", fid, rid, len(sensor), overall, (time.time() - t0) * 1000)
        return jsonify({"request_id": rid, "status": overall, "unit": "wt% Si", "results": results,
                        "context": {"si_history": si_context(hist, q), "sensor_rows_used": int(len(sensor)),
                                    "sensor_from": iso(sensor["dt"].min()), "sensor_to": iso(sensor["dt"].max())},
                        "model": {"name": fcard["name"], "furnace_id": fid, "trained_through": fcard["training_range"][1],
                                  "interval_level": fcard["interval_level"], "typical_si_range": fcard["typical_si_range"],
                                  "upset_level": fcard["upset_level"]}}), code

    # ------------------------------------------------------------------ info routes
    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": True, "model": model_meta(pipe, card.get("name", "SiPipeline"), card), "trained_through": str(pipe.train_range_[1]),
                        "n_features": len(pipe.feature_names_), "demo_model_present": os.path.exists(demo_model_path)})

    @app.route("/api/model")
    def model_info():
        body = dict(card) if card else {"name": "SiPipeline", "note": "no model card found"}
        body["required_sensor_columns"] = cfg.sensors
        return jsonify(body)

    @app.route("/api/example")
    def example():
        if not os.path.exists(example_path):
            raise ApiError(404, "no_example", "No example request file is configured")
        return Response(open(example_path).read(), mimetype="application/json")

    ui_files = ["index.html", "app.js", "app.css"]
    missing_ui = [f for f in ui_files if not os.path.exists(os.path.join(app.static_folder, f))]
    if missing_ui:
        log.warning("interface files missing from %s: %s. The API works, but the page at / will not load.",
                    app.static_folder, ", ".join(missing_ui))

    @app.route("/")
    def index():
        missing = [f for f in ui_files if not os.path.exists(os.path.join(app.static_folder, f))]
        if missing:
            page = ("<!doctype html><meta charset='utf-8'><title>Interface files missing</title>"
                    "<body style='font-family:system-ui,sans-serif;max-width:640px;margin:3rem auto;padding:0 1rem'>"
                    "<h1>The interface files are missing</h1><p>The prediction API is running, but these files were not found in "
                    f"<code>{app.static_folder}</code>:</p><ul>" + "".join(f"<li><code>{f}</code></li>" for f in missing) +
                    "</ul><p>Put <code>index.html</code>, <code>app.js</code> and <code>app.css</code> in the <code>static</code> folder next "
                    "to <code>app.py</code> and reload this page. The raw API console is at <a href='/console'>/console</a>.</p></body>")
            return Response(page, status=503, mimetype="text/html")
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/console")
    def console():
        return Response(CONSOLE_HTML, mimetype="text/html")

    # ------------------------------------------------------------------ errors and headers
    @app.errorhandler(fz.FurnaceError)
    def furnace_error(e):
        body = {"error": {"code": e.code, "message": e.message}}
        if e.details:
            body["error"]["details"] = e.details
        return jsonify(body), e.status

    @app.errorhandler(ApiError)
    def api_error(e):
        body = {"error": {"code": e.code, "message": e.message}}
        if e.details:
            body["error"]["details"] = e.details
        return jsonify(body), e.status

    @app.errorhandler(404)
    def nf(_):
        return jsonify({"error": {"code": "not_found", "message": "No such endpoint"}}), 404

    @app.errorhandler(405)
    def na(_):
        return jsonify({"error": {"code": "method_not_allowed", "message": "Method not allowed for this endpoint"}}), 405

    @app.errorhandler(413)
    def big(_):
        mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"error": {"code": "payload_too_large", "message": f"The upload is larger than {mb} MB"}}), 413

    @app.errorhandler(Exception)
    def unexpected(e):
        if isinstance(e, HTTPException):
            return jsonify({"error": {"code": e.name.lower().replace(" ", "_"), "message": e.description}}), e.code
        log.exception("unhandled error")
        return jsonify({"error": {"code": "internal_error", "message": "The server could not process this request"}}), 500

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    return app


CONSOLE_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Si prediction API console</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}textarea{width:100%;height:14rem;font-family:monospace;font-size:12px}
pre{background:#f4f4f4;padding:1rem;overflow:auto}button{padding:.5rem 1rem;margin-right:.5rem}</style></head><body>
<h1>Developer console</h1><p>Send a raw JSON body to <code>/api/predict</code>. The operator interface is at <a href="/">/</a>.</p>
<button id="ex">Load example</button><button id="go">Predict</button>
<textarea id="body" spellcheck="false"></textarea><h3>Response</h3><pre id="out">-</pre>
<script>
const $=id=>document.getElementById(id);
$('ex').onclick=async()=>{const r=await fetch('/api/example');$('body').value=r.ok?JSON.stringify(await r.json(),null,1):'No example available';};
$('go').onclick=async()=>{$('out').textContent='...';try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:$('body').value});
$('out').textContent='HTTP '+r.status+'\\n'+JSON.stringify(await r.json(),null,2);}catch(e){$('out').textContent=String(e);}};
</script></body></html>"""

app = create_app(preload_demo=True)

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG") == "1")