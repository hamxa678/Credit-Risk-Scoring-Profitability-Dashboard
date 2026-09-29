import pandas as pd
from src.models.predict import risk_band, risk_score

def test_risk_score_monotonic():
    assert risk_score(0.05) > risk_score(0.20) > risk_score(0.50)

def test_risk_band():
    assert risk_band(0.05) == "Low"
    assert risk_band(0.15) == "Medium"
    assert risk_band(0.40) == "High"
