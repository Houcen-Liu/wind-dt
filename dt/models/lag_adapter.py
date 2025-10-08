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

        # when this adapter is used in a context that does not supply
        # multi‑horizon training targets (i.e. a simple X/y pair), we
        # train a single base model instead of multiple horizon models.
        # this attribute holds that fallback model.  see `fit()` for
        # details.
        self._single_model: Optional[object] = None

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
        Fit the adapter using either a full multi‑horizon training dataset or a
        simple feature matrix and target vector.

        When called with the expected signature for lag forecasting (`Xy`
        containing lag/rolling features and multiple target columns and
        `y_cols` being a mapping of horizon name to column name), this method
        trains a separate base model for each horizon.  The resulting
        predictions will be returned as a dictionary from horizon name to
        prediction.

        If `y_cols` is not a mapping (for example when this adapter is used
        through a generic `ModelWrapper` interface, which passes a 2‑D
        feature matrix and a 1‑D target array), the method will instead
        interpret `Xy` as the feature matrix and `y_cols` as the target
        vector.  In that case a single base model is trained and stored on
        `_single_model`, and subsequent predictions will return a scalar
        rather than a horizon‑mapped dictionary.
        """

        # -- Branch: fallback for simple X/y training --
        # Accept list‑like y_cols (list/Series/ndarray) instead of a mapping.
        from collections.abc import Mapping
        if not isinstance(y_cols, Mapping):
            # Treat the first argument as the feature matrix and the second as the
            # target vector.  Xy may be a list of lists, numpy array or DataFrame.
            X = Xy
            y = y_cols

            # Determine feature order if not already set.  When called from
            # `Pipeline._make_model` the pipeline will set `feature_order` on
            # this adapter, so we respect an existing order.  Otherwise, we
            # attempt to infer it from a pandas DataFrame or fall back to
            # indexing numerically.
            if self.feature_order is None:
                if isinstance(X, pd.DataFrame):
                    self.feature_order = list(X.columns)
                else:
                    # no column names available; create a generic order based on
                    # number of features in the first row
                    try:
                        n_feats = len(X[0])
                    except Exception:
                        n_feats = 0
                    self.feature_order = [f"f{i}" for i in range(n_feats)]

            # Convert X into a DataFrame for consistent handling with base models.
            X_df = pd.DataFrame(X, columns=self.feature_order)
            y_series = pd.Series(list(y), dtype=float)
            # Filter out non‑finite targets
            mask = y_series.notna() & np.isfinite(y_series)
            if mask.sum() == 0:
                return
            model = self.base_factory()
            model.fit(X_df[mask], y_series[mask])
            self._single_model = model
            # Reset multi‑horizon models to avoid confusion on subsequent fits
            self.models.clear()
            return

        # -- Standard branch: multi‑horizon training --
        # Ensure Xy is a DataFrame and y_cols maps horizon name -> column in Xy
        if self.feature_order is None:
            # remove target columns to infer features
            non_feat = set(y_cols.values())
            self.feature_order = [c for c in Xy.columns if c not in non_feat]

        X = Xy[self.feature_order]
        for hz, yname in y_cols.items():
            # convert to float and filter out missing
            y = Xy[yname].astype(float)
            mask = y.notna() & np.isfinite(y)
            if mask.sum() < 20:
                continue
            model = self.base_factory()
            model.fit(X[mask], y[mask])
            self.models[hz] = model

    # ---- Prediction API (streaming) ----
    def predict(self, x_current: Dict[str, float]) -> Dict[str, float]:
        """
        Generate a prediction from the adapter.

        If the adapter has been trained in multi‑horizon mode (i.e. `fit()` was
        called with a mapping of horizon targets), this returns a dictionary
        mapping each horizon name to its forecast.  Predictions rely solely
        on the historical ring buffers maintained via `update_after_step()`.

        If the adapter has instead been trained in single‑horizon mode (i.e.
        `fit()` was called with a simple target vector), this will return a
        scalar prediction produced by the single base model and constructed
        from the current raw features rather than historical lag features.
        """
        # If a single fallback model is present, bypass lag feature logic and
        # produce a simple point estimate from the raw feature vector.
        if self._single_model is not None:
            # Ensure feature order is defined; default to sorted keys of x_current
            if self.feature_order is None:
                self.feature_order = sorted(list(x_current.keys()))
            # Build a 1‑row DataFrame with the expected feature order
            X_df = pd.DataFrame([[x_current.get(f, 0.0) for f in self.feature_order]],
                                columns=self.feature_order)
            # Return a float prediction to satisfy ModelWrapper contract
            return float(self._single_model.predict(X_df)[0])

        # Otherwise use historical lag/rolling features for multi‑horizon forecasts
        feats = self._make_features_from_history()
        # If no multi‑horizon models have been trained, return empty
        if not self.models:
            return {}
        # If there are no lag/roll features available yet (e.g. at the very
        # beginning of streaming), fall back to using the raw feature vector
        if not feats:
            # Ensure feature_order is known; if not, infer from x_current
            if self.feature_order is None:
                self.feature_order = sorted(list(x_current.keys()))
            X_df = pd.DataFrame([[x_current.get(f, 0.0) for f in self.feature_order]],
                                columns=self.feature_order)
            return {hz: float(m.predict(X_df)[0]) for hz, m in self.models.items()}
        # Otherwise use the lag/roll feature DataFrame
        Xdf = self._features_dataframe(feats)
        return {hz: float(m.predict(Xdf)[0]) for hz, m in self.models.items()}
