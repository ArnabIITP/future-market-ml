"""Baseline and machine-learning model factories."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def regression_models(seed: int = 17) -> dict:
    models = {
        "linear": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.8, n_jobs=-1, random_state=seed)),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = make_pipeline(SimpleImputer(strategy="median"), XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, objective="reg:squarederror", n_jobs=-1, random_state=seed))
    except ImportError:
        pass
    return models


def high_volatility_classifier(seed: int = 17):
    """Unweighted probabilistic model; caller defines threshold on training data."""
    del seed
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
