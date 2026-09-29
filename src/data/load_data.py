from pathlib import Path
import pandas as pd

def load_application_data(path: str | Path) -> pd.DataFrame:
    """Load application_train.csv or application_test.csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

def load_home_credit(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = load_application_data(data_dir / "application_train.csv")
    test_path = data_dir / "application_test.csv"
    test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    return train, test
