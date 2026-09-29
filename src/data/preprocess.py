from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLUMNS = {
    "TARGET", "SK_ID_CURR",
}

def prepare_xy(df: pd.DataFrame):
    df = df.copy()
    y = df["TARGET"].astype(int) if "TARGET" in df.columns else None
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    return X, y

def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])

    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop")
