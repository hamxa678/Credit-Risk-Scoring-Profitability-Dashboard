from __future__ import annotations
import argparse, json, os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data.load_data import load_application_data
from src.data.preprocess import prepare_xy, build_preprocessor
from src.features.build_features import add_domain_features
from src.models.evaluate import evaluate_binary

RANDOM_STATE = 42

def train_models(data_path: str, output_dir: str = "models", test_size: float = 0.2):
    df = add_domain_features(load_application_data(data_path))
    X, y = prepare_xy(df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("credit-risk-scoring")

    results = {}
    models = {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor(X_train, scale_numeric=True)),
            ("model", LogisticRegression(
                max_iter=500, class_weight="balanced", solver="liblinear",
                random_state=RANDOM_STATE
            )),
        ]),
        "xgboost": Pipeline([
            ("prep", build_preprocessor(X_train, scale_numeric=False)),
            ("model", XGBClassifier(
                n_estimators=350, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", n_jobs=4, random_state=RANDOM_STATE,
                scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
            )),
        ]),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_valid)[:, 1]
            metrics = evaluate_binary(y_valid, prob)
            mlflow.log_params({"model": name, "test_size": test_size, "random_state": RANDOM_STATE})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            path = output / f"{name}.joblib"
            joblib.dump(model, path)
            mlflow.sklearn.log_model(model, artifact_path="model")
            results[name] = metrics

    with open(output / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/application_train.csv")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    train_models(args.data, args.output)
