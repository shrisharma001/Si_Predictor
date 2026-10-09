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
    blast_off_col: str = "Fb"       # Fb <= 0 marks a shutdown / blast-off state


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
    s = sensor_df.copy()
    s[cfg.time_col] = pd.to_datetime(s[cfg.time_col])
    s = s.sort_values(cfg.time_col).drop_duplicates(cfg.time_col)
    return s.set_index(cfg.time_col)


def build_sensor_features(clean, raw, cfg, use_lags, use_roll):
    """clean/raw: hourly frames with a regular 1-hour index. Past data only."""
    f = clean[cfg.sensors].copy()
    f["blast_off"] = (pd.to_numeric(raw[cfg.blast_off_col], errors="coerce") <= 0).astype(float)
    if use_lags:
        for lag, cols in cfg.lag_spec.items():
            for c in cols:
                f[f"{c}_lag{lag}"] = clean[c].shift(lag)
    if use_roll:
        for c in cfg.roll_features:
            for w in cfg.roll_mean_windows:
                f[f"{c}_rm{w}"] = clean[c].rolling(w, min_periods=w).mean()
            for w in cfg.roll_std_windows:
                f[f"{c}_rs{w}"] = clean[c].rolling(w, min_periods=w).std()
    return f


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


MODEL_NAMES = ["mean", "persistence", "ridge", "rf", "xgb_l2", "xgb_huber"]


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
                 model="xgb_huber", seed=42):
        self.cfg = cfg or Config()
        self.use_lags, self.use_roll, self.use_history = use_lags, use_roll, use_history
        self.model_name, self.seed = model, seed

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
        self.model_ = make_model(self.model_name, self.seed)
        self.model_.fit(data[self.feature_names_], data[cfg.target])
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

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
