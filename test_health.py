"""
test_health.py - checks for the model-health (drift) report, health.py and its two routes.

  python test_health.py

Part 1 uses synthetic series with problems placed at known times, so it needs no data. Part 2 checks the routes:
the demo route needs the dataset and the demo model (skipped otherwise); the furnace route uses a small synthetic
furnace trained on the spot.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import health as H

FAILED = []


def check(ok, label, detail=""):
    print(("PASS  " if ok else "FAIL  ") + label + (f"  [{detail}]" if detail != "" else ""))
    if not ok:
        FAILED.append(label)


def synthetic(n=1500, seed=1):
    """A good model with three planted problems: narrow intervals (casts 400-550), model worse than persistence
    (800-950), a dead sensor (1100-1250)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2022-01-01", periods=n, freq="96min")
    y = 0.5 + np.cumsum(rng.normal(0, 0.01, n)) * 0.2 + rng.normal(0, 0.05, n)
    pers = np.r_[y[0], y[:-1]]
    p = y + rng.normal(0, 0.04, n)
    q = np.full(n, 0.09)
    q[400:550] = 0.02
    p[800:950] = y[800:950] + rng.normal(0, 0.15, 150)
    faults = [["Probe 3"] if 1100 <= i < 1250 else [] for i in range(n)]
    return t, y, p, p - q, p + q, pers, faults


def part1():
    print("[1] health_report on synthetic series")
    t, y, p, lo, hi, pers, faults = synthetic()
    rep = H.health_report(t, y, p, lo, hi, pers, faults)
    by = lambda k: [a for a in rep["alerts"] if a["kind"] == k]
    check({a["kind"] for a in rep["alerts"]} == {"coverage", "error", "inputs"}, "each planted problem raises its kind of alert",
          [(a["kind"], a["casts"]) for a in rep["alerts"]])

    def near(a, i0, i1, tol=60):
        return abs(pd.Timestamp(a["from"]) - t[i0]) <= tol * pd.Timedelta("96min") and abs(pd.Timestamp(a["to"]) - t[i1 - 1]) <= tol * pd.Timedelta("96min")
    check(any(near(a, 400, 550) for a in by("coverage")), "coverage alert placed where the intervals were narrow",
          [(a["from"][:10], a["to"][:10]) for a in by("coverage")])
    check(any(near(a, 800, 950) for a in by("error")), "error alert placed where the model was worse",
          [(a["from"][:10], a["to"][:10]) for a in by("error")])
    check(len(by("inputs")) == 1 and near(by("inputs")[0], 1100, 1250, tol=2) and "Probe 3 in 100 %" in by("inputs")[0]["text"],
          "sensor-fault alert covers exactly the faulty casts and names the sensor", [a["text"] for a in by("inputs")])
    check(all(a["from"] >= str(t[0])[:10] for a in rep["alerts"]), "alerts lie inside the period")
    check(rep["status"]["level"] == "ok", "the last window is healthy again", rep["status"]["text"])
    check(abs(rep["overall"]["mae"] - np.abs(y - p).mean()) < 1e-12, "overall MAE is the plain mean")
    check(sum(m["casts"] for m in rep["monthly"]) == len(t), "monthly table covers every cast")
    check(len(rep["rolling"]["time"]) <= H.MAX_POINTS, "curves are thinned for the chart", len(rep["rolling"]["time"]))

    rep2 = H.health_report(t[:300], y[:300], p[:300], lo[:300] - 1, hi[:300] + 1, pers[:300])
    check(rep2["alerts"] == [] and "fault_share" not in rep2["rolling"], "no faults given: no fault curve and no false alerts")
    bad = slice(len(t) - 120, len(t))
    p3 = p.copy(); p3[bad] = y[bad] + 0.4
    rep3 = H.health_report(t, y, p3, p3 - 0.09, p3 + 0.09, pers, faults)
    check(rep3["status"]["level"] == "alert" and "intervals too narrow" in rep3["status"]["text"]
          and "not better" in rep3["status"]["text"], "a problem in the last window shows as 'needs attention now'", rep3["status"]["text"])
    shuffled = np.random.default_rng(0).permutation(len(t))
    rep4 = H.health_report(t[shuffled], y[shuffled], p[shuffled], lo[shuffled], hi[shuffled], pers[shuffled],
                           [faults[i] for i in shuffled])
    check(rep4["alerts"] == rep["alerts"], "input order does not matter (sorted by time)")
    check(H.faulty_sensors(["previous Si result is 7.0 h old", "faulty or missing readings were replaced: Tp4, Tp8 ..."]) == ["Tp4", "Tp8"],
          "sensor names are read from prediction messages")


def part2():
    print("\n[2] routes")
    try:
        from app import create_app
    except SystemExit as e:
        print("      skipped: the built-in model is needed to start the app:", e)
        return
    root = tempfile.mkdtemp(prefix="si_health_test_")
    try:
        app = create_app(furnace_dir=root)
        c = app.test_client()
        r = c.get("/api/health/demo")
        if r.status_code == 503:
            print("      demo route skipped:", r.get_json()["error"]["message"])
        else:
            rep = r.get_json()
            check(r.status_code == 200 and rep["casts"] > 2000, "GET /api/health/demo covers the held-out period", rep.get("casts"))
            check(abs(rep["overall"]["mae"] - 0.0811) < 0.002, "its overall MAE matches the paper's test MAE", round(rep["overall"]["mae"], 4))
            jan = [a for a in rep["alerts"] if a["kind"] == "coverage" and a["from"].startswith("2016-01")]
            check(bool(jan), "the January 2016 upset is flagged as intervals too narrow")
            tp4 = [a for a in rep["alerts"] if a["kind"] == "inputs" and "Tp4" in a["text"]]
            check(bool(tp4) and tp4[0]["from"][:7] in ("2016-02", "2016-03"), "the Tp4 failure of early 2016 is flagged",
                  tp4[0]["text"] if tp4 else "")
            check(c.get("/api/health/demo").get_json() == rep, "a second request returns the same report (cached)")
        import test_furnace as TF, furnace as F, io
        s, q = TF.synthetic_furnace()
        b1, b2 = io.BytesIO(), io.BytesIO()
        s.to_csv(b1, index=False); q.to_csv(b2, index=False)
        ins = F.inspect_upload(root, b1.getvalue(), "s.csv", b2.getvalue(), "q.csv")
        fid, ctx = F.start_job(root, ins["upload_id"], "Furnace S", ins["suggested"])
        F.run_training_job(root, fid, ctx)
        r = c.get(f"/api/furnaces/{fid}/health")
        rep = r.get_json()
        card = c.get(f"/api/furnaces/{fid}").get_json()
        check(r.status_code == 200 and rep["casts"] == card["evaluation"]["test"]["n"], "GET /api/furnaces/<id>/health uses its held-out series")
        check(abs(rep["overall"]["coverage"] - card["evaluation"]["test"]["coverage"]) < 1e-9, "its coverage matches the training report")
        check(c.get("/api/furnaces/nope-000000/health").status_code == 404, "unknown furnace gives 404")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    part1()
    part2()
    print("\nRESULT:", "ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
