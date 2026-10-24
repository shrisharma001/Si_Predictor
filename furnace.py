"""
furnace.py - train and validate a silicon model on ANOTHER furnace's own history.

A model trained on one furnace does not transfer to another (different geometry, burden and instruments), so the
project treats generality as a property of the pipeline, not of a trained model. This module takes a furnace's
sensor history and Si lab results with whatever column names that plant uses, maps them onto the pipeline,
evaluates the fixed recipe with the same time-ordered protocol as the paper, and saves a model for that furnace.

  inspect_upload()      read the two uploaded files, profile their columns, suggest a column mapping
  run_training_job()    background job: map columns -> check data -> time-ordered evaluation -> final model -> save
  load_furnace()        load a saved furnace model and its card (cached)
  map_for_prediction()  apply a furnace's column mapping to newly uploaded files

Recipe (identical to the paper, nothing is tuned per furnace): sensors + rolling means/std + the 3 previous Si
results, Ridge + XGBoost-on-residuals hybrid, 90 % split-conformal intervals. Differences from our furnace:
readings are reduced to hourly medians first (plants log at different rates) and rolling statistics are built for
every selected sensor (we cannot know in advance which ones matter on another furnace).

Evaluation: samples in time order; the last 15 % are held out as a test period; the first 85 % give three
expanding-window cross-validation folds. The cleaner and the model are refitted inside every fold on that fold's
training data only. The baseline to beat is persistence (repeat the last lab result).

Storage (shared by all server workers, so a job started by one worker can be polled through another):
  <model_dir>/uploads/<upload_id>/   parsed sheets of an upload (deleted after UPLOAD_TTL_HOURS)
  <model_dir>/furnaces/<furnace_id>/ job.json (progress), pipeline.pkl and card.json once trained
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import shutil
import time
import traceback
import uuid

import numpy as np
import pandas as pd

from si_pipeline import Config, SiPipeline, prepare_sensor_frame, prepare_si_frame, require_version, valid_samples

require_version("1.4")

MIN_SAMPLES = 1000              # fewer Si results than this: refuse to train (folds would be too small)
RECOMMENDED_SAMPLES = 3000      # below this: train, but say that the estimate is rough
MAX_ROWS = 400_000              # per sheet
MAX_SHEETS = 10
MAX_SENSORS = 80
MAX_FURNACES = 20
TEST_FRACTION = 0.15
CV_BOUNDS = (0.55, 0.70, 0.85, 1.00)   # fractions of the development samples: train up to b[k], validate to b[k+1]
UPLOAD_TTL_HOURS = 6
STALE_JOB_MINUTES = 20          # a 'running' job without a heartbeat for this long was interrupted
MISSING_LIMIT = 0.5             # a sensor that is unreadable in more than half of the hours cannot be used
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,60}$")
UPLOAD_RE = re.compile(r"^[a-f0-9]{32}$")
TIME_NAME = re.compile(r"(^|[^a-z])(dt|date|time|timestamp|datetime|zeit|fecha|data)([^a-z]|$)", re.I)
SI_NAME = re.compile(r"(^|[^a-z])(si|silicon|silicium|\[si\])([^a-z]|$)", re.I)
BLAST_NAME = re.compile(r"(^fb$|blast.?flow|blast.?volume|wind.?(rate|volume|flow)|cold.?blast.?flow)", re.I)


class FurnaceError(Exception):
    def __init__(self, code, message, details=None, status=400):
        super().__init__(message)
        self.code, self.message, self.details, self.status = code, message, details, status


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    """Atomic write, so a reader in another worker never sees half a file."""
    tmp = f"{path}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ----------------------------------------------------------------------------------------------- reading files
def read_tables(raw: bytes, filename: str) -> dict:
    """All sheets of an uploaded CSV or Excel file as {sheet name: DataFrame}."""
    name = (filename or "").lower()
    try:
        if name.endswith((".csv", ".txt")):
            df = pd.read_csv(io.BytesIO(raw))
            if df.shape[1] == 1 and ";" in str(df.columns[0]):        # European exports use ';'
                df = pd.read_csv(io.BytesIO(raw), sep=";", decimal=",")
            sheets = {"(csv)": df}
        elif name.endswith((".xlsx", ".xlsm")):
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        else:
            raise FurnaceError("unsupported_file_type", f"{filename} is not a .csv or .xlsx file")
    except FurnaceError:
        raise
    except Exception:
        raise FurnaceError("unreadable_file", f"{filename} could not be read as CSV or Excel")
    sheets = {str(k): v for k, v in list(sheets.items())[:MAX_SHEETS] if len(v.columns)}
    if not sheets or all(len(v) == 0 for v in sheets.values()):
        raise FurnaceError("empty_file", f"{filename} has no rows")
    for k, v in sheets.items():
        v.columns = [str(c).strip() for c in v.columns]
        if len(v) > MAX_ROWS:
            raise FurnaceError("too_many_rows", f"{filename} (sheet {k}) has {len(v):,} rows; the limit is {MAX_ROWS:,}. "
                                                "Hourly or 10-minute data for a few years fits; resample finer data first.")
    return sheets


DATE_FORMATS = {   # tried in this order; one explicit format per column, so every row is read the same way
    "iso": ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M", "%Y-%m-%d"],
    "dayfirst": ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d-%m-%Y %H:%M:%S",
                 "%d-%m-%Y %H:%M", "%d/%m/%Y", "%d.%m.%Y"],
    "monthfirst": ["%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p", "%m-%d-%Y %H:%M",
                   "%m/%d/%Y"],
}


def _spread(s, k=20000):
    s = s.dropna().astype(str).str.strip()
    return s if len(s) <= k else s.iloc[np.linspace(0, len(s) - 1, k).astype(int)]


def _fits(sample, fmt):
    return pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean()


def detect_date_order(s: pd.Series) -> dict:
    """Which explicit formats read (almost) every value, looking across the WHOLE column (the first rows of a file
    can all have days above 12, which hides the order)."""
    sample = _spread(s)
    ok = {grp: [f for f in fmts if _fits(sample, f) >= 0.99] for grp, fmts in DATE_FORMATS.items()}
    return {"iso": bool(ok["iso"]), "dayfirst": bool(ok["dayfirst"]), "monthfirst": bool(ok["monthfirst"]),
            "formats": ok}


def parse_time(s: pd.Series, dayfirst=False) -> pd.Series:
    """Parse a time column with one explicit format chosen for the whole column; `dayfirst` decides only when both
    day-first and month-first formats fit (dates with every day <= 12)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        t = s
    else:
        order = detect_date_order(s)
        groups = ["iso", "dayfirst", "monthfirst"] if dayfirst else ["iso", "monthfirst", "dayfirst"]
        fmt = next((order["formats"][g][0] for g in groups if order["formats"][g]), None)
        txt = s.astype("string").str.strip()
        t = (pd.to_datetime(txt, format=fmt, errors="coerce") if fmt else
             pd.to_datetime(txt, errors="coerce", format="mixed", dayfirst=bool(dayfirst)))
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)                       # compare everything in UTC without a zone
    return t


def profile_sheet(df: pd.DataFrame) -> dict:
    cols = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        nonnull = s.dropna()
        info = {"name": c, "missing_share": round(float(1 - len(nonnull) / n), 3) if n else 1.0,
                "example": None if nonnull.empty else str(nonnull.iloc[0])[:40], "kind": "text"}
        if nonnull.empty:
            info["kind"] = "empty"
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["kind"] = "time"
        else:
            num = pd.to_numeric(nonnull, errors="coerce")
            if num.notna().mean() >= 0.9:
                info["kind"] = "numeric"
                info["min"], info["max"] = float(num.min()), float(num.max())
            else:
                order = detect_date_order(nonnull)
                if order["iso"] or order["dayfirst"] or order["monthfirst"]:
                    info["kind"] = "time"
                    info["dayfirst_suggested"] = bool(order["dayfirst"] and not order["monthfirst"] and not order["iso"])
                    info["dayfirst_ambiguous"] = bool(order["dayfirst"] and order["monthfirst"] and not order["iso"])
                else:
                    head = pd.to_datetime(nonnull.astype(str).head(300), errors="coerce", format="mixed")
                    if head.notna().mean() >= 0.9:          # some other layout pandas can still read
                        info["kind"] = "time"
                        info["dayfirst_suggested"] = False
                        info["dayfirst_ambiguous"] = True
        cols.append(info)
    return {"rows": int(n), "columns": cols}


def _guess_time(profile):
    times = [c["name"] for c in profile["columns"] if c["kind"] == "time"]
    named = [c for c in times if TIME_NAME.search(c)]
    return (named or times or [None])[0]


def suggest_mapping(sensor_prof: dict, si_prof: dict) -> dict:
    """A first guess the user corrects in the interface."""
    def pick_sheet(profs, want_si):
        for name, p in profs.items():
            has_si = any(SI_NAME.search(c["name"]) and c["kind"] == "numeric" for c in p["columns"])
            if has_si == want_si and _guess_time(p):
                return name
        return next(iter(profs))
    s_sheet = pick_sheet(sensor_prof, False) if len(sensor_prof) > 1 else next(iter(sensor_prof))
    q_sheet = pick_sheet(si_prof, True) if len(si_prof) > 1 else next(iter(si_prof))
    sp, qp = sensor_prof[s_sheet], si_prof[q_sheet]
    s_time, q_time = _guess_time(sp), _guess_time(qp)
    si_col = next((c["name"] for c in qp["columns"] if c["kind"] == "numeric" and SI_NAME.search(c["name"])),
                  next((c["name"] for c in qp["columns"] if c["kind"] == "numeric"), None))
    sensors = [c["name"] for c in sp["columns"] if c["kind"] == "numeric" and c["name"] != s_time
               and not SI_NAME.search(c["name"]) and c["missing_share"] < MISSING_LIMIT]
    blast = next((c for c in sensors if BLAST_NAME.search(c)), None)
    dayfirst = lambda p, col: bool(next((c.get("dayfirst_suggested", False) for c in p["columns"] if c["name"] == col), False))
    return {"sensor": {"sheet": s_sheet, "time_col": s_time, "columns": sensors[:MAX_SENSORS], "blast_off_col": blast,
                       "dayfirst": dayfirst(sp, s_time)},
            "si": {"sheet": q_sheet, "time_col": q_time, "value_col": si_col, "dayfirst": dayfirst(qp, q_time)}}


def cleanup_uploads(model_dir):
    root = os.path.join(model_dir, "uploads")
    if not os.path.isdir(root):
        return
    cutoff = time.time() - UPLOAD_TTL_HOURS * 3600
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p) and os.path.getmtime(p) < cutoff:
            shutil.rmtree(p, ignore_errors=True)


def inspect_upload(model_dir, sensor_bytes, sensor_name, si_bytes, si_name) -> dict:
    """Parse both files once, keep the parsed sheets on disk for the training job, and describe them."""
    cleanup_uploads(model_dir)
    sensor_sheets = read_tables(sensor_bytes, sensor_name)
    si_sheets = read_tables(si_bytes, si_name)
    upload_id = uuid.uuid4().hex
    d = os.path.join(model_dir, "uploads", upload_id)
    os.makedirs(d)
    meta = {"created": utcnow(), "files": {"sensor": sensor_name, "si": si_name}, "sheets": {"sensor": {}, "si": {}}}
    profiles = {"sensor": {}, "si": {}}
    for role, sheets in (("sensor", sensor_sheets), ("si", si_sheets)):
        for k, (sheet, df) in enumerate(sheets.items()):
            fn = f"{role}_{k}.pkl"                       # written and read only by this server
            df.to_pickle(os.path.join(d, fn))
            meta["sheets"][role][sheet] = fn
            profiles[role][sheet] = profile_sheet(df)
    write_json(os.path.join(d, "meta.json"), meta)
    return {"upload_id": upload_id, "files": meta["files"], "sensor_sheets": profiles["sensor"],
            "si_sheets": profiles["si"], "suggested": suggest_mapping(profiles["sensor"], profiles["si"]),
            "limits": {"min_samples": MIN_SAMPLES, "max_sensors": MAX_SENSORS}}


def load_upload(model_dir, upload_id):
    if not isinstance(upload_id, str) or not UPLOAD_RE.match(upload_id):
        raise FurnaceError("invalid_upload", "Unknown upload. Read the files again.")
    d = os.path.join(model_dir, "uploads", upload_id)
    meta = read_json(os.path.join(d, "meta.json"))
    if meta is None:
        raise FurnaceError("upload_expired", f"The uploaded files are no longer on the server (they are kept for "
                                             f"{UPLOAD_TTL_HOURS} h). Read the files again.", status=410)
    return d, meta


# ----------------------------------------------------------------------------------------------- column mapping
def safe_name(name, taken):
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_") or "sensor"
    if s[0].isdigit():
        s = "s_" + s
    base, k = s[:40], 2
    s = base
    while s in taken or s in ("dt", "Si", "blast_off") or s.startswith("si_prev") or s == "hours_since_prev":
        s, k = f"{base}_{k}", k + 1
    taken.add(s)
    return s


def validate_mapping(mapping, sensor_cols=None, si_cols=None) -> dict:
    """Check the structure of a mapping sent by the interface; return a normalised copy."""
    if not isinstance(mapping, dict):
        raise FurnaceError("invalid_mapping", "The column mapping is missing")
    s, q = mapping.get("sensor"), mapping.get("si")
    if not isinstance(s, dict) or not isinstance(q, dict):
        raise FurnaceError("invalid_mapping", "The column mapping needs a 'sensor' and an 'si' part")
    cols = s.get("columns")
    if not isinstance(cols, list) or not cols or not all(isinstance(c, str) for c in cols):
        raise FurnaceError("no_sensors", "Select at least one sensor column")
    cols = list(dict.fromkeys(cols))
    if len(cols) > MAX_SENSORS:
        raise FurnaceError("too_many_sensors", f"Select at most {MAX_SENSORS} sensor columns")
    for key, part in (("time_col", s), ("time_col", q), ("value_col", q)):
        if not isinstance(part.get(key), str) or not part.get(key):
            raise FurnaceError("invalid_mapping", f"Choose the {'Si' if key == 'value_col' else 'time'} column")
    if s["time_col"] in cols:
        raise FurnaceError("invalid_mapping", "The time column cannot also be a sensor")
    blast = s.get("blast_off_col") or None
    if blast is not None and blast not in cols:
        raise FurnaceError("invalid_mapping", "The blast-flow column must be one of the selected sensors")
    if sensor_cols is not None:
        missing = [c for c in [s["time_col"], *cols] if c not in sensor_cols]
        if missing:
            raise FurnaceError("missing_columns", "The sensor sheet does not have these columns", {"missing": missing})
    if si_cols is not None:
        missing = [c for c in [q["time_col"], q["value_col"]] if c not in si_cols]
        if missing:
            raise FurnaceError("missing_columns", "The Si sheet does not have these columns", {"missing": missing})
    taken = set()
    safe = {c: safe_name(c, taken) for c in cols}
    return {"sensor": {"sheet": s.get("sheet"), "time_col": s["time_col"], "columns": cols, "blast_off_col": blast,
                       "dayfirst": bool(s.get("dayfirst"))},
            "si": {"sheet": q.get("sheet"), "time_col": q["time_col"], "value_col": q["value_col"],
                   "dayfirst": bool(q.get("dayfirst"))},
            "safe_names": safe}


def apply_mapping(sensor_df, si_df, mapping, strict_time=True):
    """Rename a furnace's own columns onto the pipeline's names: dt, Si and one safe name per sensor.
    Returns (sensor, si, notes). Rows with an unreadable time are dropped (an error if there are many)."""
    s, q, safe = mapping["sensor"], mapping["si"], mapping["safe_names"]
    notes = []
    out = {}
    for role, df, part, cols in (("sensor", sensor_df, s, {c: safe[c] for c in s["columns"]}),
                                 ("Si", si_df, q, {q["value_col"]: "Si"})):
        missing = [c for c in [part["time_col"], *cols] if c not in df.columns]
        if missing:
            raise FurnaceError("missing_columns", f"The {role} file is missing columns this furnace's model needs",
                               {"missing": missing})
        d = df[[part["time_col"], *cols]].rename(columns={part["time_col"]: "dt", **cols})
        d["dt"] = parse_time(d["dt"], part.get("dayfirst"))
        bad = int(d["dt"].isna().sum())
        if bad:
            share = bad / max(len(d), 1)
            if strict_time and share > 0.05:
                raise FurnaceError("invalid_timestamp", f"{bad:,} of {len(d):,} rows of the {role} file have a time that "
                                                        "could not be read. Check the time column and the day-first setting.")
            notes.append(f"{bad:,} {role} rows without a readable time were skipped")
            d = d[d["dt"].notna()]
        for c in cols.values():
            d[c] = pd.to_numeric(d[c], errors="coerce")
        out[role] = d.reset_index(drop=True)
    return out["sensor"], out["Si"], notes


def make_config(mapping) -> Config:
    safe = mapping["safe_names"]
    sensors = [safe[c] for c in mapping["sensor"]["columns"]]
    blast = mapping["sensor"].get("blast_off_col")
    return Config(sensors=sensors, lag_spec={}, roll_features=list(sensors),
                  blast_off_col=safe[blast] if blast else None, resample_hourly=True)


# ----------------------------------------------------------------------------------------------- data checks
def check_data(sensor, si, cfg, display_names=None):
    """Refuse data the protocol cannot evaluate; collect warnings for the report. display_names maps the internal
    sensor names back to the furnace's own, so messages name columns the user can find."""
    show = (display_names or {}).get
    warnings = []
    if len(sensor) == 0:
        raise FurnaceError("no_sensor_data", "The sensor file has no rows with a readable time")
    raw = prepare_sensor_frame(sensor, cfg)
    si_ok = prepare_si_frame(si, cfg)
    if len(si_ok) == 0:
        raise FurnaceError("no_si_data", "The Si column has no positive numeric values")
    span = (raw.index.min(), raw.index.max())
    overlap = si_ok[(si_ok.dt > span[0]) & (si_ok.dt <= span[1])]
    if len(overlap) == 0:
        raise FurnaceError("no_overlap", f"The Si results ({si_ok.dt.min():%d %b %Y} to {si_ok.dt.max():%d %b %Y}) do not "
                                         f"overlap the sensor readings ({span[0]:%d %b %Y} to {span[1]:%d %b %Y}). "
                                         "Check that the files belong together and the day-first setting.")
    missing = raw[cfg.sensors].isna().mean()
    unusable = missing[missing > MISSING_LIMIT]
    if len(unusable) == len(cfg.sensors):
        raise FurnaceError("sensors_unusable", "None of the selected sensors has readings in most hours")
    if len(unusable):
        raise FurnaceError("sensors_unusable", "Some sensors have no reading in more than half of the hours. "
                           "Deselect them and train again.", {"sensors": [show(c, c) for c in unusable.index]})
    samples = valid_samples(si_ok, raw.index, cfg)
    samples = samples[samples.dt <= span[1]]
    if len(samples) < MIN_SAMPLES:
        raise FurnaceError("too_few_samples", f"Only {len(samples):,} Si results fall inside the sensor period (after the "
                           f"first {cfg.warmup_hours} h). At least {MIN_SAMPLES:,} are needed for a time-ordered evaluation.")
    if len(samples) < RECOMMENDED_SAMPLES:
        warnings.append(f"only {len(samples):,} Si results: the evaluation is rough; accuracy usually improves with more history")
    gaps = si_ok.dt.diff().dt.total_seconds().div(3600).dropna()
    spacing = float(gaps.median()) if len(gaps) else float("nan")
    med = float(si_ok.Si.median())
    if not 0.05 <= med <= 3.0:
        warnings.append(f"the median Si value is {med:g}; check that the column is hot-metal Si in wt%")
    hours_missing = float(raw[cfg.sensors].isna().all(axis=1).mean())
    if hours_missing > 0.05:
        warnings.append(f"no sensor reading at all in {hours_missing * 100:.0f} % of the hours; those hours are filled from earlier "
                        "readings or typical values")
    constant = [c for c in cfg.sensors if raw[c].nunique(dropna=True) <= 1]
    if constant:
        warnings.append("these sensors never change and add nothing: " + ", ".join(show(c, c) for c in constant[:8]))
    return {"samples": samples, "si_ok": si_ok, "spacing_h": spacing, "warnings": warnings,
            "sensor_span": [str(span[0]), str(span[1])], "hours": int(len(raw)), "missing_share": missing}


# ----------------------------------------------------------------------------------------------- evaluation
def make_pipeline(cfg, refit_on_all):
    return SiPipeline(cfg, use_lags=False, use_roll=True, use_history=True, model="hybrid",
                      calibrate=True, alpha=0.10, refit_on_all=refit_on_all)


def fit_until(cfg, sensor, si, end, refit_on_all=False):
    """Fit on everything recorded up to `end` (cleaner limits included), nothing later."""
    return make_pipeline(cfg, refit_on_all).fit(sensor[sensor.dt <= end], si[si.dt <= end])


def score(pipe, sensor, si_ok, query_times, train_mean):
    end = query_times.max()
    times, pred = pipe.predict(sensor[sensor.dt <= end], query_times, si_history=si_ok[si_ok.dt <= end])
    series = si_ok.set_index("dt")["Si"]
    t = pd.DatetimeIndex(times)
    y = series.loc[t].to_numpy(float)
    pers = series.shift(1).loc[t].to_numpy(float)
    pred = np.asarray(pred, float)
    ok = ~np.isnan(pers)
    t, y, pers, pred = t[ok], y[ok], pers[ok], pred[ok]
    q = float(pipe.q_)
    return {"t": t, "y": y, "pred": pred, "pers": pers, "mean": np.full(len(y), train_mean), "q": q}


def metrics(y, p):
    e = np.abs(y - p)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"mae": float(e.mean()), "r2": float(1 - ((y - p) ** 2).sum() / ss_tot) if ss_tot > 0 else None,
            "within_0_1": float((e <= 0.1).mean())}


def block_bootstrap(e_a, e_b, block=50, n_boot=2000, seed=0):
    """Paired block bootstrap of the mean difference in absolute error (respects autocorrelation)."""
    d = np.asarray(e_a) - np.asarray(e_b)
    n = len(d)
    block = int(min(block, max(5, n // 10)))
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci": [float(lo), float(hi)], "significant": bool(hi < 0 or lo > 0),
            "block": block, "resamples": n_boot}


def summarise(sc):
    m = metrics(sc["y"], sc["pred"])
    return {"n": int(len(sc["y"])), "from": str(sc["t"].min()), "to": str(sc["t"].max()),
            "model": m, "persistence": metrics(sc["y"], sc["pers"]), "mean": metrics(sc["y"], sc["mean"]),
            "coverage": float((np.abs(sc["y"] - sc["pred"]) <= sc["q"]).mean()), "interval_width": 2 * sc["q"]}


def verdict(test, boot):
    m, p = test["model"]["mae"], test["persistence"]["mae"]
    gain = 1 - m / p if p > 0 else 0.0
    if m < p and boot["significant"]:
        level, text = "good", (f"On the held-out period the model's average error is {gain * 100:.0f} % lower than simply repeating "
                               "the last lab result, and the difference is statistically significant.")
    elif m < p:
        level, text = "weak", (f"On the held-out period the model's average error is {gain * 100:.0f} % lower than repeating the last "
                               "lab result, but the difference is not statistically significant. Treat it as no better.")
    else:
        level, text = "poor", ("On the held-out period the model is not better than simply repeating the last lab result. "
                               "Do not rely on it; more history or other sensors may help.")
    cov = test["coverage"]
    if cov < 0.85:
        text += f" Its 90 % intervals held the measured value only {cov * 100:.0f} % of the time, so they are too narrow here."
    return {"level": level, "text": text, "gain_vs_persistence": gain}


# ----------------------------------------------------------------------------------------------- the job
class Job:
    STEPS = ["Reading files", "Checking the data", "Cross-validation fold 1 of 3", "Cross-validation fold 2 of 3",
             "Cross-validation fold 3 of 3", "Held-out test period", "Training the final model", "Saving"]

    def __init__(self, job_dir):
        self.path = os.path.join(job_dir, "job.json")
        self.state = read_json(self.path, {})

    def update(self, **kw):
        self.state.update(kw, updated=utcnow())
        write_json(self.path, self.state)

    def step(self, i):
        self.update(status="running", step=self.STEPS[i], step_index=i, steps=len(self.STEPS), progress=round(i / len(self.STEPS), 3))


def furnace_dir(model_dir, furnace_id):
    if not isinstance(furnace_id, str) or not ID_RE.match(furnace_id):
        raise FurnaceError("invalid_furnace", "Unknown furnace", status=404)
    return os.path.join(model_dir, "furnaces", furnace_id)


def new_furnace_id(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "furnace"
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def job_status(model_dir, furnace_id):
    """Status from job.json; a 'running' job whose heartbeat stopped is reported as interrupted."""
    d = furnace_dir(model_dir, furnace_id)
    job = read_json(os.path.join(d, "job.json"))
    if job is None:
        raise FurnaceError("unknown_job", "No training job for this furnace", status=404)
    if job.get("status") in ("queued", "running"):
        upd = pd.Timestamp(job.get("updated") or job.get("created"))
        if (pd.Timestamp.now(tz="UTC") - upd) > pd.Timedelta(minutes=STALE_JOB_MINUTES):
            job.update(status="failed", error={"code": "interrupted", "message": "Training stopped unexpectedly "
                                               "(the server was restarted). Start it again."})
    return job


def acquire_training_lock(model_dir, furnace_id):
    """One training job at a time across all workers (training is CPU-heavy)."""
    root = os.path.join(model_dir, "furnaces")
    os.makedirs(root, exist_ok=True)
    lock = os.path.join(root, ".training.lock")
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, furnace_id.encode())
            os.close(fd)
            return lock
        except FileExistsError:
            holder = open(lock).read().strip() if os.path.exists(lock) else ""
            try:
                st = job_status(model_dir, holder)["status"] if holder else "failed"
            except FurnaceError:
                st = "failed"
            if st in ("queued", "running"):
                raise FurnaceError("busy", "Another furnace model is being trained. Wait until it finishes.", status=409)
            try:
                os.remove(lock)                     # left behind by an interrupted job
            except FileNotFoundError:
                pass
    raise FurnaceError("busy", "Another furnace model is being trained. Wait until it finishes.", status=409)


def start_job(model_dir, upload_id, name, mapping):
    """Validate the request and create the job directory. The caller runs run_training_job(...) in a thread."""
    name = (name or "").strip()
    if not 1 <= len(name) <= 60:
        raise FurnaceError("invalid_name", "Give the furnace a name of 1 to 60 characters")
    up_dir, meta = load_upload(model_dir, upload_id)
    m = mapping if isinstance(mapping, dict) else {}
    s_sheet = (m.get("sensor") or {}).get("sheet") or next(iter(meta["sheets"]["sensor"]))
    q_sheet = (m.get("si") or {}).get("sheet") or next(iter(meta["sheets"]["si"]))
    if s_sheet not in meta["sheets"]["sensor"] or q_sheet not in meta["sheets"]["si"]:
        raise FurnaceError("invalid_mapping", "Unknown sheet")
    s_cols = pd.read_pickle(os.path.join(up_dir, meta["sheets"]["sensor"][s_sheet])).columns
    q_cols = pd.read_pickle(os.path.join(up_dir, meta["sheets"]["si"][q_sheet])).columns
    m = dict(m, sensor=dict(m.get("sensor") or {}, sheet=s_sheet), si=dict(m.get("si") or {}, sheet=q_sheet))
    norm = validate_mapping(m, list(s_cols), list(q_cols))
    existing = list_furnaces(model_dir)
    if len(existing) >= MAX_FURNACES:
        raise FurnaceError("too_many_furnaces", f"At most {MAX_FURNACES} furnace models can be kept. Delete one first.")
    fid = new_furnace_id(name)
    lock = acquire_training_lock(model_dir, fid)
    d = furnace_dir(model_dir, fid)
    os.makedirs(d)
    write_json(os.path.join(d, "job.json"), {"furnace_id": fid, "name": name, "status": "queued", "created": utcnow(),
                                             "updated": utcnow(), "progress": 0.0, "step": "Waiting to start"})
    return fid, {"up_dir": up_dir, "meta": meta, "mapping": norm, "name": name, "lock": lock}


def run_training_job(model_dir, furnace_id, ctx):
    d = furnace_dir(model_dir, furnace_id)
    job = Job(d)
    t_start = time.time()
    try:
        job.step(0)
        mapping, meta, up = ctx["mapping"], ctx["meta"], ctx["up_dir"]
        sensor_raw = pd.read_pickle(os.path.join(up, meta["sheets"]["sensor"][mapping["sensor"]["sheet"]]))
        si_raw = pd.read_pickle(os.path.join(up, meta["sheets"]["si"][mapping["si"]["sheet"]]))
        sensor, si, notes = apply_mapping(sensor_raw, si_raw, mapping)
        del sensor_raw, si_raw
        cfg = make_config(mapping)

        job.step(1)
        chk = check_data(sensor, si, cfg, {v: k for k, v in mapping["safe_names"].items()})
        samples, si_ok = chk["samples"], chk["si_ok"]
        times = samples.dt.reset_index(drop=True)
        n = len(times)
        n_dev = int(n * (1 - TEST_FRACTION))
        bounds = [int(b * n_dev) for b in CV_BOUNDS]

        folds = []
        for k in range(3):
            job.step(2 + k)
            train_end, val = times.iloc[bounds[k] - 1], times.iloc[bounds[k]:bounds[k + 1]]
            pipe = fit_until(cfg, sensor, si_ok, train_end)
            sc = score(pipe, sensor, si_ok, val, float(si_ok[si_ok.dt <= train_end].Si.mean()))
            folds.append(dict(summarise(sc), fold=k + 1, train_samples=int(pipe.n_train_)))

        job.step(5)
        dev_end, test_t = times.iloc[n_dev - 1], times.iloc[n_dev:]
        pipe = fit_until(cfg, sensor, si_ok, dev_end)
        sc = score(pipe, sensor, si_ok, test_t, float(si_ok[si_ok.dt <= dev_end].Si.mean()))
        test = dict(summarise(sc), train_samples=int(pipe.n_train_))
        boot = block_bootstrap(np.abs(sc["y"] - sc["pred"]), np.abs(sc["y"] - sc["pers"]))
        series = {"time": [t.isoformat() for t in sc["t"]], "actual": np.round(sc["y"], 4).tolist(),
                  "prediction": np.round(sc["pred"], 4).tolist(), "persistence": np.round(sc["pers"], 4).tolist(),
                  "lower": np.round(np.maximum(sc["pred"] - sc["q"], 0), 4).tolist(),
                  "upper": np.round(sc["pred"] + sc["q"], 4).tolist()}

        job.step(6)
        final = fit_until(cfg, sensor, si_ok, times.iloc[-1], refit_on_all=True)

        job.step(7)
        si_train = si_ok[si_ok.dt <= times.iloc[-1]].Si
        upset = round(float(si_train.quantile(0.995)), 2)
        inv = {v: k for k, v in mapping["safe_names"].items()}
        faults = final.cleaner_.fault_rate_.sort_values(ascending=False)
        card = {
            "furnace_id": furnace_id, "name": ctx["name"], "created_utc": utcnow(), "kind": "furnace",
            "source_files": meta["files"],
            "mapping": mapping,
            "required_columns": {"sensor": [mapping["sensor"]["time_col"], *mapping["sensor"]["columns"]],
                                 "si": [mapping["si"]["time_col"], mapping["si"]["value_col"]]},
            "recipe": "sensors (hourly medians) + rolling mean/std of every sensor + 3 previous Si results; Ridge + "
                      "XGBoost on residuals; 90 % split-conformal intervals (same fixed settings as the paper)",
            "training_samples": int(final.n_train_), "calibration_samples": int(final.n_cal_),
            "training_range": [str(final.train_range_[0]), str(final.train_range_[1])],
            "sensor_span": chk["sensor_span"], "n_features": len(final.feature_names_),
            "median_spacing_h": round(chk["spacing_h"], 2),
            "typical_si_range": [round(float(si_train.quantile(0.10)), 3), round(float(si_train.quantile(0.90)), 3)],
            "upset_level": upset, "stale_hours": round(max(6.0, 4 * chk["spacing_h"]), 1),
            "interval_level": 0.9, "interval_half_width": round(final.q_, 4),
            "sensor_faults": [{"sensor": inv.get(c, c), "share": round(float(v), 4)} for c, v in faults.items() if v > 0.001][:8],
            "evaluation": {"protocol": f"time-ordered: last {TEST_FRACTION:.0%} of samples held out; 3 expanding-window "
                                       "folds on the rest; cleaner and model refitted on each fold's training data",
                           "folds": folds, "test": test, "test_vs_persistence": boot, "verdict": verdict(test, boot)},
            "warnings": notes + chk["warnings"],
            "training_seconds": round(time.time() - t_start, 1),
        }
        final.save(os.path.join(d, "pipeline.pkl"))
        write_json(os.path.join(d, "test_series.json"), series)
        write_json(os.path.join(d, "card.json"), card)
        job.update(status="done", progress=1.0, step="Done", finished=utcnow())
    except FurnaceError as e:
        job.update(status="failed", error={"code": e.code, "message": e.message, "details": e.details})
    except Exception as e:                                       # never leave a job 'running'
        traceback.print_exc()
        job.update(status="failed", error={"code": "internal_error", "message": f"Training failed: {type(e).__name__}"})
    finally:
        try:
            os.remove(ctx["lock"])
        except OSError:
            pass


# ----------------------------------------------------------------------------------------------- using a model
_cache = {}


def list_furnaces(model_dir):
    root = os.path.join(model_dir, "furnaces")
    out = []
    if not os.path.isdir(root):
        return out
    for fid in sorted(os.listdir(root)):
        if not ID_RE.match(fid):
            continue
        card = read_json(os.path.join(root, fid, "card.json"))
        if card:
            ev = card["evaluation"]
            out.append({"furnace_id": fid, "name": card["name"], "created_utc": card["created_utc"],
                        "training_range": card["training_range"], "verdict": ev["verdict"]["level"],
                        "test_mae": ev["test"]["model"]["mae"], "persistence_mae": ev["test"]["persistence"]["mae"],
                        "required_columns": card["required_columns"]})
        else:
            try:
                job = job_status(model_dir, fid)
            except FurnaceError:
                continue
            out.append({"furnace_id": fid, "name": job.get("name"), "status": job.get("status"),
                        "created_utc": job.get("created")})
    return out


def load_furnace(model_dir, furnace_id):
    d = furnace_dir(model_dir, furnace_id)
    p, c = os.path.join(d, "pipeline.pkl"), os.path.join(d, "card.json")
    if not (os.path.exists(p) and os.path.exists(c)):
        raise FurnaceError("unknown_furnace", "No trained model for this furnace", status=404)
    key = (furnace_id, os.path.getmtime(p))
    if key not in _cache:
        _cache.clear() if len(_cache) > 8 else None
        _cache[key] = (SiPipeline.load(p), read_json(c))
    return _cache[key]


def delete_furnace(model_dir, furnace_id):
    d = furnace_dir(model_dir, furnace_id)
    if not os.path.isdir(d):
        raise FurnaceError("unknown_furnace", "No such furnace", status=404)
    job = read_json(os.path.join(d, "job.json"), {})
    if job.get("status") in ("queued", "running") and job_status(model_dir, furnace_id)["status"] == "running":
        raise FurnaceError("busy", "This furnace is still training", status=409)
    shutil.rmtree(d)


def translate_messages(msg, card):
    """Show the furnace's own sensor names in warnings instead of the internal safe names."""
    inv = {v: k for k, v in card["mapping"]["safe_names"].items()}
    m = re.match(r"^(faulty or missing readings were replaced: |outside the range seen in training \(clipped\): )(.*)$", msg)
    if not m:
        return msg
    names = [inv.get(x, x) for x in m.group(2).replace(" ...", "").split(", ")]
    return m.group(1) + ", ".join(names) + (" ..." if msg.endswith(" ...") else "")
