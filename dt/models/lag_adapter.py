from __future__ import annotations
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Callable, Optional
import numpy as np
import pandas as pd

from .base import ModelWrapper

# ---------- Config types ----------
@dataclass
class LagSpec:
    # how many timesteps each horizon looks ahead (in rows)
    horizons: Dict[str, int]     # e.g. {"H1": 6, "D1": 144, "W1": 1008, "M1": 4320}
    # raw lags to include (in rows) per base signal
    lags: Dict[str, List[int]]   # e.g. {"v":[1,6,12], "y":[1,6,12]}
    # rolling windows (rows) per base signal -> stats to compute
    rolls: Dict[str, List[int]]  # e.g. {"v":[6,36], "y":[6,36]}
    roll_stats: List[str] = None # subset of ["mean","std","min","max"]

    def __post_init__(self):
        if self.roll_stats is None:
            self.roll_stats = ["mean","std"]


class LagFeatureForecastAdapter(ModelWrapper):
    """
    Wraps any sklearn-like regressor into a multi-horizon forecaster
    that uses ONLY current + historical inputs. One model per horizon.

    base_factory: function -> sklearn estimator (fresh instance)
      e.g. for LGBM:   lambda: LGBMRegressor(**params)
           for SVR:    lambda: Pipeline([('scaler', StandardScaler()), ('svr', SVR(**p))])

    base_inputs: list of raw input names the pipeline passes in x (e.g. ['v','y','wdir_sin','wdir_cos','temp'])
    step_per_hour: rows per hour (10-min data -> 6)
    """
    def __init__(
        self,
        base_factory: Callable[[], object],
        lag_spec: LagSpec,
        base_inputs: List[str],
        step_per_hour: int = 6,
    ):
        self.base_factory = base_factory
        self.lag_spec = lag_spec
        self.base_inputs = base_inputs
        self.step_per_hour = step_per_hour

        # ring buffers (history) for each base input
        self.hist: Dict[str, deque] = {k: deque(maxlen=max(
            [max(lag_spec.lags.get(k, [0])+lag_spec.rolls.get(k, [1]))*2, 6]
        )) for k in base_inputs}

        # trained models per horizon
        self.models: Dict[str, object] = {}
        # fixed feature order after first fit
        self.feature_order: Optional[List[str]] = None

    # ---- History management (call this each tick with current x, y if available) ----
    def update_after_step(self, x: Dict[str, float], y_actual: Optional[float] = None):
        # store y into history if you include it in base_inputs
        for k in self.base_inputs:
            val = x.get(k) if k != "y" else (y_actual if y_actual is not None else x.get("y"))
            self.hist[k].append(np.nan if val is None else float(val))

    # ---- Feature engineering from history only ----
    def _make_features_from_history(self) -> Dict[str, float]:
        feats = {}
        for nm, q in self.hist.items():
            if not q:
                continue
            s = pd.Series(q, dtype=float)
            # raw lags
            for lag in self.lag_spec.lags.get(nm, []):
                if len(s) > lag:
                    feats[f"{nm}_lag{lag}"] = float(s.iloc[-lag-1])
            # rolling stats
            for w in self.lag_spec.rolls.get(nm, []):
                if len(s) >= w:
                    sw = s.iloc[-w:]
                    if "mean" in self.lag_spec.roll_stats: feats[f"{nm}_mean{w}"] = float(sw.mean())
                    if "std"  in self.lag_spec.roll_stats: feats[f"{nm}_std{w}"]  = float(sw.std(ddof=0))
                    if "min"  in self.lag_spec.roll_stats: feats[f"{nm}_min{w}"]  = float(sw.min())
                    if "max"  in self.lag_spec.roll_stats: feats[f"{nm}_max{w}"]  = float(sw.max())
        return feats

    def _features_dataframe(self, feats: Dict[str, float]) -> pd.DataFrame:
        if self.feature_order is None:
            self.feature_order = sorted(list(feats.keys()))
        return pd.DataFrame([[feats.get(f, 0.0) for f in self.feature_order]], columns=self.feature_order)

    # ---- Training API (batch) ----
    def fit(self, Xy: pd.DataFrame, y_cols: Dict[str, str]):
        """
        Xy: dataframe that already contains base inputs timeline AND the lag/rolling features computed historically,
            plus one column per horizon target (window-mean or point-ahead), named by y_cols.
        y_cols: mapping horizon->column name, e.g. {"H1":"y_H1", "D1":"y_D1", ...}
        """
        # infer feature order from provided columns if missing
        if self.feature_order is None:
            non_feat = set(y_cols.values())
            self.feature_order = [c for c in Xy.columns if c not in non_feat]

        X = Xy[self.feature_order]
        for hz, yname in y_cols.items():
            y = Xy[yname].astype(float)
            mask = y.notna() & np.isfinite(y)
            if mask.sum() < 20:
                continue
            model = self.base_factory()
            model.fit(X[mask], y[mask])
            self.models[hz] = model

    # ---- Prediction API (streaming) ----
    def predict(self, x_current: Dict[str, float]) -> Dict[str, float]:
        # NOTE: prediction uses ONLY history; x_current is not strictly needed but kept for interface parity
        feats = self._make_features_from_history()
        if not feats or not self.models:
            return {}
        Xdf = self._features_dataframe(feats)
        return {hz: float(m.predict(Xdf)[0]) for hz, m in self.models.items()}
