from __future__ import annotations
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def fit_segments(customer_features: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)),
    ])
    labels = pipe.fit_predict(customer_features)
    return pipe, labels

def save_segments(model, path: str):
    joblib.dump(model, path)
