import pandas as pd
from src.features.build_features import add_domain_features

def test_domain_features():
    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [100000],
        "AMT_CREDIT": [200000],
        "AMT_ANNUITY": [10000],
        "DAYS_BIRTH": [-3650],
    })
    out = add_domain_features(df)
    assert "CREDIT_TO_INCOME" in out
    assert "AGE_YEARS" in out
    assert abs(out.loc[0, "CREDIT_TO_INCOME"] - 2.0) < 1e-9
