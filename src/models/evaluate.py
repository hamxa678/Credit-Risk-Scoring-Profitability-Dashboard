from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, brier_score_loss
)

def ks_statistic(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y = y_true[order]
    pos = max((y == 1).sum(), 1)
    neg = max((y == 0).sum(), 1)
    tpr = np.cumsum(y == 1) / pos
    fpr = np.cumsum(y == 0) / neg
    return float(np.max(np.abs(tpr - fpr)))

def evaluate_binary(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": int(confusion_matrix(y_true, y_pred).ravel()[0]),
        "fp": int(confusion_matrix(y_true, y_pred).ravel()[1]),
        "fn": int(confusion_matrix(y_true, y_pred).ravel()[2]),
        "tp": int(confusion_matrix(y_true, y_pred).ravel()[3]),
    }
