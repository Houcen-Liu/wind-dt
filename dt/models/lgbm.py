
import lightgbm as lgb
import numpy as np
import pandas as pd
from .base import ModelWrapper

class LGBMRegressorWrapper(ModelWrapper):
    def __init__(self, monotone_speed_feature=True, feature_order=None):
        self.feature_order = feature_order or []
        mono = [(+1 if (monotone_speed_feature and f in ("v","v2","v3")) else 0)
                for f in self.feature_order]
        self.model = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=6,
            min_child_samples=80, subsample=0.9, colsample_bytree=0.9,
            monotone_constraints=mono
        )

    def fit(self, X, y):
        # Ensure named columns during fit
        X_df = pd.DataFrame(X, columns=self.feature_order)
        self.model.fit(X_df, y)

    def predict(self, x):
        # Ensure named columns during predict
        X_df = pd.DataFrame([[x.get(f, 0.0) for f in self.feature_order]],
                            columns=self.feature_order)
        return float(self.model.predict(X_df)[0])
