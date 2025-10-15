from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Dict

try:
    from .base import ModelWrapper
except Exception:
    try:
        from dt.models.base import ModelWrapper
    except Exception:
        import os, sys
        CUR = os.path.dirname(os.path.abspath(__file__))
        PARENT = os.path.dirname(CUR)
        for path in (CUR, PARENT):
            if path not in sys.path:
                sys.path.insert(0, path)
        from base import ModelWrapper


class layar_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        m = max(1, dvalues.shape[0])
        self.dweights = np.dot(self.inputs.T, dvalues) / m
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) / m
        self.dinputs = np.dot(dvalues, self.weights.T)


class activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0.0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0.0] = 0.0


class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues


class Loss_MSE:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2, axis=-1)
    def backward(self, y_pred, y_true):
        self.dinputs = 2.0 * (y_pred - y_true)


class BaseNN:
    def __init__(self, input_dim: int, hidden: int = 32, lr: float = 1e-3, clip: float = 1.0):
        self.dense1 = layar_dense(input_dim, hidden)
        self.act1 = activation_ReLU()
        self.dense2 = layar_dense(hidden, 1)
        self.act2 = Activation_Linear()
        self.loss = Loss_MSE()
        self.lr = lr
        self.clip = clip
    def _step(self, Xb, yb):
        self.dense1.forward(Xb)
        self.act1.forward(self.dense1.output)
        self.dense2.forward(self.act1.output)
        self.act2.forward(self.dense2.output)
        y_pred = self.act2.output
        _ = self.loss.forward(y_pred, yb)
        self.loss.backward(y_pred, yb)
        self.act2.backward(self.loss.dinputs)
        self.dense2.backward(self.act2.dinputs)
        self.act1.backward(self.dense2.dinputs)
        self.dense1.backward(self.act1.dinputs)
        for layer in (self.dense1, self.dense2):
            np.clip(layer.dweights, -self.clip, self.clip, out=layer.dweights)
            np.clip(layer.dbiases, -self.clip, self.clip, out=layer.dbiases)
            layer.weights -= self.lr * layer.dweights
            layer.biases  -= self.lr * layer.dbiases
        return y_pred
    def fit_batch(self, X, y, epochs=1, batch_size=32, shuffle=True):
        n = X.shape[0]
        if n == 0:
            return
        for _ in range(epochs):
            idx = np.arange(n)
            if shuffle:
                np.random.shuffle(idx)
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                self._step(X[idx[s:e]], y[idx[s:e]])
    def partial_fit_one(self, x_row, y_scalar):
        Xb = x_row.reshape(1, -1)
        yb = np.array([[y_scalar]], dtype=np.float64)
        self._step(Xb, yb)
    def predict_one(self, x_row):
        Xb = x_row.reshape(1, -1)
        self.dense1.forward(Xb)
        self.act1.forward(self.dense1.output)
        self.dense2.forward(self.act1.output)
        self.act2.forward(self.dense2.output)
        return float(self.act2.output.ravel()[0])
    def predict(self, X):
        self.dense1.forward(X)
        self.act1.forward(self.dense1.output)
        self.dense2.forward(self.act1.output)
        self.act2.forward(self.dense2.output)
        return self.act2.output.ravel()


class NNWrapper(ModelWrapper):
    def __init__(self,
                 feature_order: Optional[Sequence[str]] = None,
                 kernel: str = "rbf",
                 C: float = 10.0,
                 epsilon: float = 0.1,
                 gamma = "scale",
                 coef0: float = 0.0,
                 horizons: Sequence[int] = (1, 24, 168, 720),
                 hidden: int = 32,
                 lr: float = 1e-3,
                 clip: float = 1.0,
                 normalize: bool = True):
        self.feature_order = list(feature_order or [])
        self.normalize = normalize
        self.horizons = list(horizons)
        self.models: Dict[int, BaseNN] = {}
        self.X_mean = None
        self.X_std  = None
        self.y_mean = {h: 0.0 for h in self.horizons}
        self.y_std  = {h: 1.0 for h in self.horizons}
        self.hidden = hidden
        self.lr = lr
        self.clip = clip
    def _ensure_models(self, input_dim: int):
        if not self.models:
            for h in self.horizons:
                self.models[h] = BaseNN(input_dim=input_dim, hidden=self.hidden, lr=self.lr, clip=self.clip)
    def _fit_normalizer(self, X: np.ndarray, y: np.ndarray, horizon: int):
        if self.normalize:
            if self.X_mean is None:
                self.X_mean = X.mean(axis=0, keepdims=True)
                self.X_std  = X.std(axis=0, keepdims=True) + 1e-8
            self.y_mean[horizon] = float(np.nanmean(y))
            self.y_std[horizon]  = float(np.nanstd(y) + 1e-8)
        else:
            if self.X_mean is None:
                self.X_mean = np.zeros((1, X.shape[1]))
                self.X_std  = np.ones((1, X.shape[1]))
            self.y_mean[horizon] = 0.0
            self.y_std[horizon]  = 1.0
    def _normX(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / self.X_std
    def _normy(self, y: np.ndarray, h: int) -> np.ndarray:
        return (y - self.y_mean[h]) / self.y_std[h]
    def _denormy(self, yhat: np.ndarray, h: int) -> np.ndarray:
        return yhat * self.y_std[h] + self.y_mean[h]
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            if not self.feature_order:
                self.feature_order = list(X.columns)
            X_arr = X[self.feature_order].to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if not self.feature_order:
                self.feature_order = [f"f{i}" for i in range(X_arr.shape[1])]
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if X_arr.shape[0] == 0:
            raise ValueError("empty X passed to fit")
        self._ensure_models(X_arr.shape[1])
        h = self.horizons[0]
        self._fit_normalizer(X_arr, y_arr, h)
        Xn = self._normX(X_arr)
        yn = self._normy(y_arr, h)
        self.models[h].fit_batch(Xn, yn, epochs=1, batch_size=32, shuffle=True)
    def _steps_map(self, step_minutes: int) -> Dict[int, int]:
        return {h: max(1, int(round((h * 60) / step_minutes))) for h in self.horizons}
    def _make_shifted_targets(self, y: np.ndarray, timestamps, step_minutes: int) -> Dict[int, np.ndarray]:
        if isinstance(timestamps, pd.Series):
            _ = timestamps.to_numpy()
        y = y.reshape(-1, 1).astype(np.float64)
        steps_map = self._steps_map(step_minutes)
        out = {}
        for h, steps in steps_map.items():
            y_shift = np.roll(y, -steps, axis=0)
            y_shift[-steps:, 0] = np.nan
            out[h] = y_shift
        return out
    def fit_multi(self, X, y, timestamps, step_minutes: int = 60, epochs: int = 1, batch_size: int = 32):
        if isinstance(X, pd.DataFrame):
            if not self.feature_order:
                self.feature_order = list(X.columns)
            X_arr = X[self.feature_order].to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if not self.feature_order:
                self.feature_order = [f"f{i}" for i in range(X_arr.shape[1])]
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n = X_arr.shape[0]
        if n == 0:
            raise ValueError("empty X passed to fit_multi")
        steps_map = self._steps_map(step_minutes)
        feasible = [h for h, s in steps_map.items() if s < n]
        if not feasible:
            raise ValueError("no feasible horizons for current data length; increase rows or reduce step_minutes")
        self._ensure_models(X_arr.shape[1])
        y_map = self._make_shifted_targets(y_arr, timestamps, step_minutes)
        mask = np.ones(n, dtype=bool)
        for h in feasible:
            mask &= ~np.isnan(y_map[h]).ravel()
        if not mask.any():
            raise ValueError("mask removed all rows; provide more data or adjust horizons/step_minutes")
        Xc = X_arr[mask]
        for h in list(y_map.keys()):
            y_map[h] = y_map[h][mask]
        if self.X_mean is None:
            self._fit_normalizer(Xc, y_map[feasible[0]], feasible[0])
        Xn = self._normX(Xc)
        for h in feasible:
            self._fit_normalizer(Xc, y_map[h], h)
            yn = self._normy(y_map[h], h)
            self.models[h].fit_batch(Xn, yn, epochs=epochs, batch_size=batch_size, shuffle=True)
        return self
    def partial_fit(self, x: dict, y_true_map: Optional[Dict[int, float]] = None, y_true: Optional[float] = None):
        if not self.feature_order:
            raise ValueError("feature_order empty")
        X_df = pd.DataFrame([[x.get(f, 0.0) for f in self.feature_order]], columns=self.feature_order)
        xr = X_df.to_numpy(dtype=np.float64)
        if self.X_mean is None:
            self.X_mean = np.zeros((1, xr.shape[1]))
            self.X_std  = np.ones((1, xr.shape[1]))
        Xn = self._normX(xr).reshape(-1)
        if y_true_map:
            for h, val in y_true_map.items():
                if val is None or np.isnan(val):
                    continue
                if h not in self.models:
                    self.models[h] = BaseNN(input_dim=xr.shape[1], hidden=self.hidden, lr=self.lr, clip=self.clip)
                yv = np.array([[val]], dtype=np.float64)
                self._fit_normalizer(xr, yv, h)
                yn = self._normy(yv, h).ravel()[0]
                self.models[h].partial_fit_one(Xn, yn)
        elif y_true is not None:
            h = self.horizons[0]
            yv = np.array([[y_true]], dtype=np.float64)
            self._fit_normalizer(xr, yv, h)
            yn = self._normy(yv, h).ravel()[0]
            self.models[h].partial_fit_one(Xn, yn)
    def predict(self, x: dict) -> float:
        if not self.feature_order:
            raise ValueError("feature_order empty")
        X_df = pd.DataFrame([[x.get(f, 0.0) for f in self.feature_order]], columns=self.feature_order)
        xr = X_df.to_numpy(dtype=np.float64)
        if self.X_mean is None:
            self.X_mean = np.zeros((1, xr.shape[1]))
            self.X_std  = np.ones((1, xr.shape[1]))
        Xn = self._normX(xr).reshape(-1)
        h = self.horizons[0]
        if h not in self.models:
            self.models[h] = BaseNN(input_dim=xr.shape[1], hidden=self.hidden, lr=self.lr, clip=self.clip)
        yhat_n = self.models[h].predict_one(Xn)
        yhat = self._denormy(np.array([yhat_n]), h)[0]
        return float(yhat)
    def predict_horizons(self, x: dict) -> Dict[int, float]:
        if not self.feature_order:
            raise ValueError("feature_order empty")
        X_df = pd.DataFrame([[x.get(f, 0.0) for f in self.feature_order]], columns=self.feature_order)
        xr = X_df.to_numpy(dtype=np.float64)
        if self.X_mean is None:
            self.X_mean = np.zeros((1, xr.shape[1]))
            self.X_std  = np.ones((1, xr.shape[1]))
        Xn = ((xr - self.X_mean) / self.X_std).reshape(-1)
        out = {}
        for h in self.horizons:
            if h not in self.models:
                self.models[h] = BaseNN(input_dim=xr.shape[1], hidden=self.hidden, lr=self.lr, clip=self.clip)
            yhat_n = self.models[h].predict_one(Xn)
            yhat   = yhat_n * self.y_std.get(h, 1.0) + self.y_mean.get(h, 0.0)
            out[h] = float(yhat)
        return out
