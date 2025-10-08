from __future__ import annotations
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import ModelWrapper

class SVRWrapper(ModelWrapper):
    """
    Sklearn SVM for Regression with an internal StandardScaler.
    - Keeps feature names consistent by using DataFrames for fit/predict.
    - Options (via __init__):
        kernel: "rbf" | "linear" | "poly" | "sigmoid"  (default "rbf")
        C: float (default 10.0)
        epsilon: float (default 0.1)
        gamma: "scale" | "auto" | float (default "scale")
        degree: int (for poly kernel; default 3)
        coef0: float (for poly/sigmoid; default 0.0)
    """
    def __init__(
        self,
        feature_order=None,
        kernel: str = "rbf",
        C: float = 10.0,        # If underfitting (too smooth): raise C to 50–100.
        epsilon: float = 0.1,   # If noisy: increase epsilon (0.2–0.4) to ignore small residuals.
        gamma = "scale",        # If predictions blow up at edges: reduce C and/or use "auto" gamma (stronger smoothing).
        coef0: float = 0.0,
    ):
        self.feature_order = feature_order or []
        self.model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, coef0=coef0)),
        ])

    def fit(self, X, y):
        # Ensure named columns for stability and to mirror predict
        X_df = pd.DataFrame(X, columns=self.feature_order)
        self.model.fit(X_df, y)

    def predict(self, x: dict) -> float:
        X_df = pd.DataFrame([[x.get(f, 0.0) for f in self.feature_order]],
                            columns=self.feature_order)
        return float(self.model.predict(X_df)[0])

    # SVR doesn't natively do quantiles; keep base default (None)
