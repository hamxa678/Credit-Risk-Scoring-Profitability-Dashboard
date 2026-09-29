import numpy as np
from src.models.evaluate import ks_statistic, evaluate_binary

def test_ks_range():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.9])
    assert 0 <= ks_statistic(y, p) <= 1

def test_metrics_exist():
    metrics = evaluate_binary([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert "roc_auc" in metrics
    assert "ks" in metrics
