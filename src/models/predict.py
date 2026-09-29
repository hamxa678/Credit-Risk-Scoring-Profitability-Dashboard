from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from src.features.build_features import add_domain_features

def load_model(path: str | Path):
    return joblib.load(path)

def predict_pd(model, df: pd.DataFrame):
    X = add_domain_features(df.copy())
    return model.predict_proba(X)[:, 1]

def risk_score(pd_value: float) -> int:
    """Simple monotonic score: 900 = lower risk, 300 = higher risk."""
    return int(max(300, min(900, round(900 - 600 * pd_value))))

def risk_band(pd_value: float) -> str:
    if pd_value < 0.10:
        return "Low"
    if pd_value < 0.25:
        return "Medium"
    return "High"
