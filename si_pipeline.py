"""
si_pipeline.py
==============
One place for everything between raw sensor data and a silicon (Si) prediction.

The SAME functions are used for training, evaluation and inference, so the
model never sees data that was prepared differently from its training data.

Stages
------
1. SensorCleaner    : fault codes / spikes -> NaN, forward-fill (past only),
                      winsorise, median-fill.  Limits are learned on TRAIN only.
2. build_sensor_features : base sensors, lags, rolling statistics (past only).
3. history_features : previous Si lab results (strictly earlier samples only).
4. make_model       : Ridge / RandomForest / XGBoost (L2 or Huber) / baselines.
5. SiPipeline       : bundles 1-4 into one object you can joblib.dump().
"""
from __future__ import annotations

__version__ = "1.4"   # 1.4: optional hourly resampling and blast-off column (other furnaces, furnace.py)
                      # 1.3: hybrid model, conformal intervals, predict_frame (needed by train_final, app, tests)

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def require_version(minimum):
    """Stop with a clear message if this file is older than the calling script needs."""
    have = tuple(int(x) for x in __version__.split("."))
    need = tuple(int(x) for x in minimum.split("."))
    if have < need:
        raise SystemExit(f"si_pipeline.py is version {__version__} but this script needs {minimum} or newer. "
                         "Copy the latest si_pipeline.py from the project folder over yours.")


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SENSORS = [
    "Fb", "Ph", "Pc", "Tc", "Fo", "dP", "dPu", "dPl", "Pt", "Th", "CO2", "H2",
    "Tt1", "Tt2", "Tt3", "Tt4",
    "Tp1", "Tp2", "Tp3", "Tp4", "Tp5", "Tp6", "Tp7", "Tp8", "Tp9", "Tp10",
    "R",
]


@dataclass
class Config:
    time_col: str = "dt"
    target: str = "Si"
    sensors: list = field(default_factory=lambda: list(SENSORS))

    # lag features carried over from the existing lag analysis
    lag_spec: dict = field(default_factory=lambda: {
        1: ["Fb", "Pt", "dP", "Tp3", "Tp6", "Tp5", "CO2", "H2"],
        6: ["Th"],
        8: ["R"],
        12: ["CO2"],
    })

    # rolling statistics (window ends at the current hour, uses the past only)
    roll_features: list = field(default_factory=lambda: [
        "Fb", "Pt", "dP", "Th", "CO2", "H2", "Tp3", "Tp5", "Tp6", "Tp8", "R",
    ])
    roll_mean_windows: tuple = (3, 6, 12, 24)
    roll_std_windows: tuple = (6, 24)

    n_history: int = 3              # number of previous Si lab results used
    align_tolerance_hours: float = 1.0
    warmup_hours: int = 24          # drop samples without enough sensor history
    blast_off_col: str | None = "Fb"   # Fb <= 0 marks a shutdown / blast-off state; None = no such column
    resample_hourly: bool = False   # True: reduce readings to hourly medians first (other furnaces; see prepare_sensor_frame)


# ----------------------------------------------------------------------------
# 1. CLEANING
# ----------------------------------------------------------------------------
class SensorCleaner:
    """Learns what a 'bad' value looks like from TRAINING data, then masks it.

    A value is treated as a sensor fault if
      * it is a repeated extreme value far from the bulk (e.g. 3311.20, -35.56),
      * it lies beyond an IQR fence (k * IQR) on the high side, or is a
        NEGATIVE value beyond the fence on the low side.
    Zero is never treated as a fault (Fb = 0 is a real shutdown state).

    Faults become NaN -> forward-filled for up to `ffill_limit` hours (this only
    looks backward in time) -> winsorised -> any remaining NaN = train median.
    """

    def __init__(self, fence_k=10.0, sentinel_k=3.0, min_sentinel_count=5,
                 winsor=(0.001, 0.999), ffill_limit=6):
        self.fence_k = fence_k
        self.sentinel_k = sentinel_k
        self.min_sentinel_count = min_sentinel_count
        self.winsor = winsor
        self.ffill_limit = ffill_limit

    def _numeric(self, df):
        return df[self.cols_].apply(pd.to_numeric, errors="coerce").astype(float)

    def _fault_mask(self, x):
        bad = pd.DataFrame(False, index=x.index, columns=x.columns)
        for c in self.cols_:
            lo_f, hi_f = self.fence_[c]
            v = x[c]
            m = v.isna() | v.isin(self.sentinels_[c]) | (v > hi_f) | ((v < lo_f) & (v < 0))
            bad[c] = m
        return bad

    def fit(self, df, cols):
        self.cols_ = list(cols)
        x = self._numeric(df)
        self.sentinels_, self.fence_ = {}, {}
        for c in self.cols_:
            v = x[c].dropna()
            q1, q3 = v.quantile([0.25, 0.75])
            q05, q95 = v.quantile([0.05, 0.95])
            # robust scale: IQR can collapse for set-point-controlled signals
            # (e.g. Pt sits exactly at 1.40 half the time), so use a floor
            iqr = max(q3 - q1, 0.5 * (q95 - q05))
            if iqr > 0:
                self.fence_[c] = (q1 - self.fence_k * iqr, q3 + self.fence_k * iqr)
            else:
                self.fence_[c] = (-np.inf, np.inf)
            counts = v.value_counts()
            sent = []
            for val in (v.min(), v.max()):
                far = (val < q1 - self.sentinel_k * iqr) or (val > q3 + self.sentinel_k * iqr)
                if val != 0 and counts.get(val, 0) >= self.min_sentinel_count and far and iqr > 0:
                    sent.append(float(val))
            self.sentinels_[c] = sorted(set(sent))
        bad = self._fault_mask(x)
        clean = x.mask(bad)
        self.lo_ = clean.quantile(self.winsor[0])
        self.hi_ = clean.quantile(self.winsor[1])
        self.median_ = clean.median()
        self.fault_counts_ = bad.sum()
        self.fault_rate_ = bad.mean()
        return self

    def transform(self, df):
        x = self._numeric(df)
        x = x.mask(self._fault_mask(x))
        x = x.ffill(limit=self.ffill_limit)
        x = x.clip(lower=self.lo_, upper=self.hi_, axis=1)
        return x.fillna(self.median_)

    def report(self):
        r = pd.DataFrame({
            "faults": self.fault_counts_,
            "fault_%": (self.fault_rate_ * 100).round(2),
            "sentinels": pd.Series(self.sentinels_).apply(lambda s: ", ".join(f"{a:g}" for a in s)),
        })
        return r[r.faults > 0]


# ----------------------------------------------------------------------------
# 2. SENSOR FEATURES (hourly frame indexed by time)
# ----------------------------------------------------------------------------
def prepare_sensor_frame(sensor_df, cfg):
    """Time-indexed sensor frame. With cfg.resample_hourly the readings are reduced to hourly medians on a regular grid.
    Each bin is labelled by its END and closed on the right, so the row at 10:00 holds readings from (09:00, 10:00]
    and never anything recorded after the query time. Empty hours become NaN and are handled by the cleaner."""
    s = sensor_df.copy()
    s[cfg.time_col] = pd.to_datetime(s[cfg.time_col])
    s = s.sort_values(cfg.time_col).drop_duplicates(cfg.time_col)
    s = s.set_index(cfg.time_col)
    if getattr(cfg, "resample_hourly", False):
        num = s[cfg.sensors].apply(pd.to_numeric, errors="coerce")
        s = num.resample("1h", label="right", closed="right").median()
    return s


def build_sensor_features(clean, raw, cfg, use_lags, use_roll):
    """clean/raw: hourly frames with a regular 1-hour index. Past data only."""
    cols = {c: clean[c] for c in cfg.sensors}
    if cfg.blast_off_col:
        cols["blast_off"] = (pd.to_numeric(raw[cfg.blast_off_col], errors="coerce") <= 0).astype(float)
    if use_lags:
        for lag, names in cfg.lag_spec.items():
            for c in names:
                cols[f"{c}_lag{lag}"] = clean[c].shift(lag)
    if use_roll:
        for c in cfg.roll_features:
            for w in cfg.roll_mean_windows:
                cols[f"{c}_rm{w}"] = clean[c].rolling(w, min_periods=w).mean()
            for w in cfg.roll_std_windows:
                cols[f"{c}_rs{w}"] = clean[c].rolling(w, min_periods=w).std()
    return pd.DataFrame(cols, index=clean.index)       # built in one go (same column order as before)


# ----------------------------------------------------------------------------
# 3. SI HISTORY FEATURES  (identical code path for training and inference)
# ----------------------------------------------------------------------------
def prepare_si_frame(si_df, cfg):
    d = si_df[[cfg.time_col, cfg.target]].copy()
    d[cfg.time_col] = pd.to_datetime(d[cfg.time_col])
    d[cfg.target] = pd.to_numeric(d[cfg.target], errors="coerce")
    d = d.dropna()
    d = d[d[cfg.target] > 0]                      # Si <= 0 is not a valid lab result
    return d.sort_values(cfg.time_col).drop_duplicates(cfg.time_col).reset_index(drop=True)


def history_features(query_times, si_times, si_vals, n):
    """For each query time, use only Si results from STRICTLY EARLIER samples."""
    q = pd.DatetimeIndex(query_times).values
    st = pd.DatetimeIndex(si_times).values
    idx = np.searchsorted(st, q, side="left") - 1
    out = {}
    cols = []
    for k in range(1, n + 1):
        j = idx - (k - 1)
        out[f"si_prev{k}"] = np.where(j >= 0, si_vals[np.clip(j, 0, None)], np.nan)
        cols.append(f"si_prev{k}")
    h = pd.DataFrame(out)
    h["si_prev_mean"] = h[cols].mean(axis=1)
    last_t = np.where(idx >= 0, st[np.clip(idx, 0, None)], np.datetime64("NaT"))
    h["hours_since_prev"] = (q - last_t) / np.timedelta64(1, "h")
    return h


# ----------------------------------------------------------------------------
# DATASET ASSEMBLY
# ----------------------------------------------------------------------------
def valid_samples(si, sensor_index, cfg):
    """The row universe shared by every experiment (so results are comparable)."""
    t0 = sensor_index.min() + pd.Timedelta(hours=cfg.warmup_hours)
    keep = si[cfg.time_col] >= t0
    keep &= np.arange(len(si)) >= cfg.n_history       # need n previous results
    return si[keep].reset_index(drop=True)


def assemble(sensor_feats, samples, si_all, cfg, use_history):
    """Align each Si sample with the latest sensor row at or before its time."""
    left = samples[[cfg.time_col, cfg.target]].sort_values(cfg.time_col)
    right = sensor_feats.reset_index().rename(columns={sensor_feats.index.name or "index": cfg.time_col})
    m = pd.merge_asof(left, right, on=cfg.time_col, direction="backward",
                      tolerance=pd.Timedelta(hours=cfg.align_tolerance_hours))
    if use_history:
        h = history_features(m[cfg.time_col], si_all[cfg.time_col], si_all[cfg.target].values,
                             cfg.n_history)
        m = pd.concat([m, h], axis=1)
    return m.dropna().reset_index(drop=True)


def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile of the calibration scores."""
    sc = np.sort(np.asarray(scores, float))
    k = min(int(np.ceil((len(sc) + 1) * (1 - alpha))), len(sc))
    return float(sc[k - 1])


def assemble_keep(sensor_feats, queries, si_all, cfg, use_history):
    """Like assemble(), but never drops a row: missing inputs stay NaN so the caller can explain why."""
    left = queries[[cfg.time_col, cfg.target]].sort_values(cfg.time_col)
    right = sensor_feats.reset_index().rename(columns={sensor_feats.index.name or "index": cfg.time_col})
    m = pd.merge_asof(left, right, on=cfg.time_col, direction="backward",
                      tolerance=pd.Timedelta(hours=cfg.align_tolerance_hours))
    if use_history:
        h = history_features(m[cfg.time_col], si_all[cfg.time_col], si_all[cfg.target].values, cfg.n_history)
        m = pd.concat([m.reset_index(drop=True), h], axis=1)
    return m


# ----------------------------------------------------------------------------
# 4. MODELS
# ----------------------------------------------------------------------------
class Persistence:
    """Predict the previous Si lab result."""
    def fit(self, X, y): return self
    def predict(self, X): return X["si_prev1"].values


class XGBEarlyStop:
    """XGBoost with early stopping on the LAST 15% (chronologically) of training."""
    def __init__(self, objective="reg:squarederror", huber_slope=0.15, seed=42):
        self.objective, self.huber_slope, self.seed = objective, huber_slope, seed

    def _make(self, n_estimators, base_score, early=None):
        kw = dict(n_estimators=n_estimators, learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
                  reg_lambda=5.0, objective=self.objective, base_score=base_score,
                  random_state=self.seed, n_jobs=-1)
        if self.objective == "reg:pseudohubererror":
            kw["huber_slope"] = self.huber_slope
        if early:
            kw["early_stopping_rounds"] = early
        return XGBRegressor(**kw)

    def fit(self, X, y):
        n = len(X)
        cut = int(n * 0.85)
        base = float(np.median(y))
        m = self._make(2000, base, early=50)
        m.fit(X.iloc[:cut], y.iloc[:cut], eval_set=[(X.iloc[cut:], y.iloc[cut:])], verbose=False)
        self.best_n_ = int(m.best_iteration + 1)
        self.model_ = self._make(max(50, int(self.best_n_ * 1.1)), base)
        self.model_.fit(X, y, verbose=False)
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, X):
        return self.model_.predict(X)


class HybridModel:
    """Primary model: Ridge, then XGBoost fitted to the Ridge residuals (same recipe as run_peaks.Hybrid)."""
    def __init__(self, seed=42):
        self.lin = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        self.res = XGBEarlyStop("reg:squarederror", seed=seed)

    def fit(self, X, y):
        self.lin.fit(X, y)
        self.res.fit(X, y - self.lin.predict(X))
        return self

    def predict(self, X):
        return self.lin.predict(X) + self.res.predict(X)


MODEL_NAMES = ["mean", "persistence", "ridge", "rf", "xgb_l2", "xgb_huber", "hybrid"]


def make_model(name, seed=42):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "persistence":
        return Persistence()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "rf":
        return RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.33,
                                     n_jobs=-1, random_state=seed)
    if name == "xgb_l2":
        return XGBEarlyStop("reg:squarederror", seed=seed)
    if name == "xgb_huber":
        return XGBEarlyStop("reg:pseudohubererror", seed=seed)
    if name == "hybrid":
        return HybridModel(seed)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------
def evaluate(y, p, normal_range=(0.2, 1.2)):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(y - p)
    nm = (y >= normal_range[0]) & (y <= normal_range[1])
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p),
        "within_0.05": float((err <= 0.05).mean()),
        "within_0.10": float((err <= 0.10).mean()),
        "R2_normal": r2_score(y[nm], p[nm]) if nm.sum() > 2 else np.nan,
        "MAE_normal": mean_absolute_error(y[nm], p[nm]) if nm.sum() > 2 else np.nan,
        "n": int(len(y)),
    }


# ----------------------------------------------------------------------------
# 5. THE PIPELINE OBJECT  (what you save and what the Flask app will load)
# ----------------------------------------------------------------------------
class SiPipeline:
    def __init__(self, cfg=None, use_lags=True, use_roll=True, use_history=True,
                 model="xgb_huber", seed=42, calibrate=False, alpha=0.10, refit_on_all=True):
        """calibrate=True adds split-conformal intervals (nominal level 1-alpha): the model is first fitted on the
        first 85 % of the training samples and the residuals of the last 15 % set the half-width. With
        refit_on_all=True the final model is then refitted on all samples (more data, slightly conservative
        intervals); with False the calibrated model itself is kept (exactly the recipe in run_uncertainty.py)."""
        self.cfg = cfg or Config()
        self.use_lags, self.use_roll, self.use_history = use_lags, use_roll, use_history
        self.model_name, self.seed = model, seed
        self.calibrate, self.alpha, self.refit_on_all = calibrate, alpha, refit_on_all

    # -- training ----------------------------------------------------------
    def fit(self, sensor_df, si_df):
        cfg = self.cfg
        raw = prepare_sensor_frame(sensor_df, cfg)
        si = prepare_si_frame(si_df, cfg)
        samples = valid_samples(si, raw.index, cfg)
        train_end = samples[cfg.time_col].max()
        self.cleaner_ = SensorCleaner().fit(raw.loc[:train_end], cfg.sensors)
        clean = self.cleaner_.transform(raw)
        feats = build_sensor_features(clean, raw, cfg, self.use_lags, self.use_roll)
        data = assemble(feats, samples, si, cfg, self.use_history)
        self.feature_names_ = [c for c in data.columns if c not in (cfg.time_col, cfg.target)]
        X, y = data[self.feature_names_], data[cfg.target]
        self.q_, self.n_cal_ = None, 0
        if self.calibrate:
            cut = int(len(data) * 0.85)
            m = make_model(self.model_name, self.seed).fit(X.iloc[:cut], y.iloc[:cut])
            res = np.abs(y.iloc[cut:].values - m.predict(X.iloc[cut:]))
            self.q_, self.n_cal_ = conformal_quantile(res, self.alpha), len(res)
            self.model_ = make_model(self.model_name, self.seed).fit(X, y) if self.refit_on_all else m
        else:
            self.model_ = make_model(self.model_name, self.seed).fit(X, y)
        self.n_train_ = len(data)
        self.train_range_ = (data[cfg.time_col].min(), data[cfg.time_col].max())
        return self

    # -- inference ---------------------------------------------------------
    def predict(self, sensor_df, query_times, si_history=None):
        """sensor_df : hourly readings covering at least the last `warmup_hours`
                       before each query time.
        query_times  : when the prediction is wanted (one per cast).
        si_history   : DataFrame[dt, Si] of PAST lab results (needed if use_history)."""
        cfg = self.cfg
        raw = prepare_sensor_frame(sensor_df, cfg)
        clean = self.cleaner_.transform(raw)
        feats = build_sensor_features(clean, raw, cfg, self.use_lags, self.use_roll)
        q = pd.DataFrame({cfg.time_col: pd.to_datetime(query_times), cfg.target: 0.0})
        if self.use_history:
            if si_history is None:
                raise ValueError("this pipeline needs si_history (previous Si lab results)")
            si_all = prepare_si_frame(si_history, cfg)
        else:
            si_all = None
        data = assemble(feats, q, si_all, cfg, self.use_history)
        return data[cfg.time_col].values, self.model_.predict(data[self.feature_names_])

    def predict_frame(self, sensor_df, query_times, si_history=None, stale_hours=6.0, upset_level=1.2):
        """Production-style prediction. Returns one row per query with the prediction, an interval (if the pipeline
        was calibrated), a status ('ok' / 'warning' / 'no_prediction') and human-readable messages.
        Raises ValueError only for unusable inputs (missing columns, no history frame)."""
        cfg = self.cfg
        missing = sorted((set(cfg.sensors) | {cfg.time_col}) - set(sensor_df.columns))
        if missing:
            raise ValueError(f"sensor_df is missing required columns: {missing}")
        if len(sensor_df) == 0:
            raise ValueError("sensor_df is empty")
        if self.use_history and si_history is None:
            raise ValueError("si_history (previous Si lab results with columns dt, Si) is required")
        raw = prepare_sensor_frame(sensor_df, cfg)
        numeric = raw[cfg.sensors].apply(pd.to_numeric, errors="coerce").astype(float)
        fault = self.cleaner_._fault_mask(numeric)
        clean = self.cleaner_.transform(raw)
        feats = build_sensor_features(clean, raw, cfg, self.use_lags, self.use_roll)
        q = pd.DataFrame({cfg.time_col: pd.to_datetime(query_times), cfg.target: 0.0})
        si_all = prepare_si_frame(si_history, cfg) if self.use_history else None
        d = assemble_keep(feats, q, si_all, cfg, self.use_history)
        ok = ~d[self.feature_names_].isna().any(axis=1).values
        pred = np.full(len(d), np.nan)
        if ok.any():
            pred[ok] = self.model_.predict(d.loc[ok, self.feature_names_])
        rows = []
        for i in range(len(d)):
            t = d[cfg.time_col].iloc[i]
            msgs = []
            pos = raw.index.searchsorted(t, side="right") - 1
            if not ok[i]:
                if pos < 0 or (t - raw.index[pos]) > pd.Timedelta(hours=cfg.align_tolerance_hours):
                    msgs.append(f"no sensor reading within {cfg.align_tolerance_hours:g} h before the query")
                elif self.use_history and d[[f"si_prev{k}" for k in range(1, cfg.n_history + 1)]].iloc[i].isna().any():
                    msgs.append(f"fewer than {cfg.n_history} previous Si results are available")
                else:
                    msgs.append(f"not enough sensor history (need about {cfg.warmup_hours} hourly rows before the query)")
                rows.append((t, np.nan, np.nan, np.nan, "no_prediction", "; ".join(msgs)))
                continue
            if pos >= 0:
                f_row = fault.iloc[pos]
                bad = [c for c in cfg.sensors if f_row[c]]
                if bad:
                    msgs.append("faulty or missing readings were replaced: " + ", ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""))
                vals = numeric.iloc[pos]
                beyond = [c for c in cfg.sensors if not f_row[c] and not np.isnan(vals[c])
                          and (vals[c] < self.cleaner_.lo_[c] or vals[c] > self.cleaner_.hi_[c])]
                if beyond:
                    msgs.append("outside the range seen in training (clipped): " + ", ".join(beyond[:6]))
                if cfg.blast_off_col and vals[cfg.blast_off_col] <= 0:
                    msgs.append("blast appears to be off (shutdown state)")
            if self.use_history:
                if d["hours_since_prev"].iloc[i] > stale_hours:
                    msgs.append(f"previous Si result is {d['hours_since_prev'].iloc[i]:.1f} h old; intervals are less reliable after long gaps")
                if d["si_prev1"].iloc[i] > upset_level:
                    msgs.append(f"previous Si above {upset_level}: predictions and intervals are unreliable during upsets")
            lo = hi = np.nan
            if self.q_ is not None:
                lo, hi = max(pred[i] - self.q_, 0.0), pred[i] + self.q_
            rows.append((t, pred[i], lo, hi, "warning" if msgs else "ok", "; ".join(msgs)))
        out = pd.DataFrame(rows, columns=["time", "prediction", "lower", "upper", "status", "messages"])
        out["interval_level"] = (1 - self.alpha) if self.q_ is not None else np.nan
        return out

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)