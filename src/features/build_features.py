from __future__ import annotations
import numpy as np
import pandas as pd

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact, leakage-safe financial/demographic features."""
    out = df.copy()

    if "AMT_INCOME_TOTAL" in out and "AMT_CREDIT" in out:
        out["CREDIT_TO_INCOME"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)

    if "AMT_ANNUITY" in out and "AMT_INCOME_TOTAL" in out:
        out["ANNUITY_TO_INCOME"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)

    if "AMT_CREDIT" in out and "AMT_GOODS_PRICE" in out:
        out["CREDIT_TO_GOODS_PRICE"] = out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"].replace(0, np.nan)

    if "DAYS_BIRTH" in out:
        out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25

    if "DAYS_EMPLOYED" in out:
        emp = out["DAYS_EMPLOYED"].replace(365243, np.nan)
        out["EMPLOYED_YEARS"] = -emp / 365.25

    return out.replace([np.inf, -np.inf], np.nan)

def make_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c for c in [
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
            "AGE_YEARS", "EMPLOYED_YEARS",
            "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
        ] if c in df.columns
    ]
    out = df[cols].copy()
    return out
