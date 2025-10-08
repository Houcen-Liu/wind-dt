from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def lgbm_factory(params=None):
    p = dict(n_estimators=800, learning_rate=0.05, max_depth=6,
             min_child_samples=80, subsample=0.9, colsample_bytree=0.9)
    if params: p.update(params)
    return LGBMRegressor(**p)

def svr_factory(params=None):
    p = dict(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    if params: p.update(params)
    return Pipeline([("scaler", StandardScaler()), ("svr", SVR(**p))])
