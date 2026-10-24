"""
health.py - is the model still trustworthy? Drift checks on predictions that have since been checked against the lab.

In operation every prediction is followed, an hour or two later, by the measured Si. Comparing the two over a
moving window shows when the model stops earning its place:

  * error      rolling MAE of the model against rolling MAE of persistence (repeating the last lab result);
               the model is only useful while it is the lower of the two
  * coverage   share of measured values inside the 90 % interval; well below 90 % means the intervals are too
               narrow for the current conditions (the paper found this in upsets and after long gaps)
  * inputs     share of casts whose latest sensor row had readings replaced by the cleaner (a failing
               instrument shows up here before it shows up in the error)

health_report() turns one series of checked predictions into rolling curves, alert periods and a monthly table.
The interface runs it on the held-out period of the demo model and on the held-out period of each trained
furnace model; a live deployment would run it on its own prediction log in the same way.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

WINDOW = 100            # casts in the moving window (about a week at one result every 1.6 h)
MIN_ALERT_CASTS = 25    # shorter excursions are noise, not alerts
COVERAGE_ALERT = 0.80   # nominal 0.90
FAULT_ALERT = 0.10      # share of casts with replaced readings
MAX_POINTS = 600        # rolling curves are thinned to this many points for the chart
FAULT_RE = re.compile(r"^faulty or missing readings were replaced: (.*)$")


def faulty_sensors(messages):
    """Sensor names from a prediction's 'faulty or missing readings were replaced: a, b ...' message."""
    for m in messages or []:
        hit = FAULT_RE.match(m)
        if hit:
            return [s for s in hit.group(1).replace(" ...", "").split(", ") if s]
    return []


def _periods(flag, times, min_len):
    """Contiguous runs of True at least min_len long -> [(start index, end index)]."""
    out, start = [], None
    for i, f in enumerate(list(flag) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def pct(x):
    return f"{x * 100:.0f} %"


def _thin(n, k=MAX_POINTS):
    return np.arange(n) if n <= k else np.unique(np.linspace(0, n - 1, k).astype(int))


def health_report(times, actual, prediction, lower, upper, persistence, faults=None, window=WINDOW, level=0.9):
    """times, actual, prediction, lower, upper, persistence: equal-length sequences for predictions that were made
    (no NaN). faults: optional list (one entry per cast) of the sensors whose readings were replaced."""
    t = pd.DatetimeIndex(pd.to_datetime(times))
    y, p = np.asarray(actual, float), np.asarray(prediction, float)
    lo, hi, pers = np.asarray(lower, float), np.asarray(upper, float), np.asarray(persistence, float)
    n = len(t)
    if n == 0:
        raise ValueError("no checked predictions")
    order = np.argsort(t.values, kind="stable")
    t, y, p, lo, hi, pers = t[order], y[order], p[order], lo[order], hi[order], pers[order]
    faults = [faults[i] for i in order] if faults is not None else None
    w = int(min(window, max(10, n // 4)))
    e_m, e_p = pd.Series(np.abs(y - p)), pd.Series(np.abs(y - pers))
    inside = pd.Series(((y >= lo) & (y <= hi)).astype(float))
    has_fault = pd.Series([1.0 if f else 0.0 for f in faults]) if faults is not None else None
    # Looking back over a finished period, curves and alert periods use a CENTRED window (the w casts around each
    # point), so a problem is drawn where it happened. The current status below uses the trailing window, which is
    # what a live system can compute.
    cen = lambda s: s.rolling(w, min_periods=w // 2, center=True).mean()
    r_m, r_p, r_c = cen(e_m), cen(e_p), cen(inside)
    r_f = cen(has_fault) if has_fault is not None else None

    alerts = []
    def add(kind, flag, text_fn, affected=None):
        """affected: per-cast booleans; the alert is trimmed to its first and last affected cast, because a centred
        window already 'sees' a problem up to w/2 casts before it starts."""
        for a, b in _periods(flag.fillna(False).to_numpy(bool), t, MIN_ALERT_CASTS):
            if affected is not None:
                hit = np.flatnonzero(affected[a:b + 1])
                if len(hit):
                    a, b = a + hit[0], a + hit[-1]
            alerts.append({"kind": kind, "from": t[a].isoformat(), "to": t[b].isoformat(), "casts": int(b - a + 1),
                           "text": text_fn(a, b)})

    add("coverage", r_c < COVERAGE_ALERT,
        lambda a, b: f"Intervals held only {pct(inside[a:b + 1].mean())} of measured values (target {pct(level)}); "
                     f"the highest Si in this period was {y[a:b + 1].max():.2f} wt%.", affected=(inside.to_numpy() == 0))
    def error_text(a, b):
        lo_i, hi_i = max(0, a - w // 2), min(n, b + w // 2 + 1)     # the casts the centred windows averaged over
        return (f"Around this period the model's average error ({e_m[lo_i:hi_i].mean():.3f} wt% over {hi_i - lo_i} casts) "
                f"was above that of repeating the last lab result ({e_p[lo_i:hi_i].mean():.3f}).")
    add("error", r_m > r_p, error_text)
    if has_fault is not None:
        def fault_text(a, b):
            m = b - a + 1
            cnt = Counter(s for f in faults[a:b + 1] for s in f)
            top = ", ".join(f"{s} in {pct(c / m)}" for s, c in cnt.most_common(3) if c / m >= 0.02)
            return f"Sensor readings were replaced in {pct(has_fault[a:b + 1].mean())} of casts" + (f" ({top})." if top else ".")
        add("inputs", r_f > FAULT_ALERT, fault_text, affected=(has_fault.to_numpy() > 0))
    alerts.sort(key=lambda x: x["from"])

    df = pd.DataFrame({"t": t, "e_m": e_m.values, "e_p": e_p.values, "inside": inside.values, "y": y})
    if has_fault is not None:
        df["fault"] = has_fault.values
    monthly = []
    for per, g in df.groupby(df.t.dt.to_period("M")):
        row = {"month": str(per), "casts": int(len(g)), "mae": float(g.e_m.mean()), "persistence_mae": float(g.e_p.mean()),
               "coverage": float(g.inside.mean()), "max_si": float(g.y.max())}
        if has_fault is not None:
            row["fault_share"] = float(g.fault.mean())
        monthly.append(row)

    k = _thin(n)
    keep = ~np.isnan(r_m.to_numpy()[k])
    k = k[keep]
    rnd = lambda s: [None if np.isnan(v) else round(float(v), 4) for v in s]
    rolling = {"time": [t[i].isoformat() for i in k], "mae": rnd(r_m.to_numpy()[k]), "persistence_mae": rnd(r_p.to_numpy()[k]),
               "coverage": rnd(r_c.to_numpy()[k])}
    if r_f is not None:
        rolling["fault_share"] = rnd(r_f.to_numpy()[k])

    last = slice(max(0, n - w), n)
    now = {"mae": float(e_m[last].mean()), "persistence_mae": float(e_p[last].mean()), "coverage": float(inside[last].mean())}
    if has_fault is not None:
        now["fault_share"] = float(has_fault[last].mean())
    problems = []
    if now["coverage"] < COVERAGE_ALERT:
        problems.append("intervals too narrow")
    if now["mae"] > now["persistence_mae"]:
        problems.append("not better than repeating the last result")
    if now.get("fault_share", 0) > FAULT_ALERT:
        problems.append("frequent sensor faults")
    status = {"level": "alert" if problems else "ok",
              "text": (f"Over the last {w} casts: " + "; ".join(problems) + ".") if problems else
                      f"Over the last {w} casts the model beats repeating the last result and its intervals hold "
                      f"{pct(now['coverage'])} of measured values."}
    return {"window": w, "casts": n, "from": t[0].isoformat(), "to": t[-1].isoformat(), "level": level,
            "overall": {"mae": float(e_m.mean()), "persistence_mae": float(e_p.mean()), "coverage": float(inside.mean())},
            "latest": now, "status": status, "alerts": alerts, "monthly": monthly, "rolling": rolling,
            "thresholds": {"coverage": COVERAGE_ALERT, "fault_share": FAULT_ALERT, "min_alert_casts": MIN_ALERT_CASTS}}
