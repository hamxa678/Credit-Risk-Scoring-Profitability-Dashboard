# Credit Risk Scoring & Profitability Dashboard

End-to-end credit-risk ML project using the **Home Credit Default Risk** dataset.

## Features

- Logistic Regression baseline
- XGBoost default prediction
- Class-imbalance handling
- ROC-AUC, KS, precision, recall, F1 and Brier score
- Financial/domain feature engineering
- Customer segmentation with K-Means
- Probability-based risk score and risk bands
- Expected loss and expected profitability
- Interactive Streamlit dashboard
- MLflow experiment tracking
- Pytest test suite
- Ruff linting
- Docker + Docker Compose
- GitHub Actions CI
- Jupyter notebooks

## Dataset

Download the Home Credit Default Risk dataset from Kaggle:

https://www.kaggle.com/competitions/home-credit-default-risk/data

At minimum, place:

```text
data/raw/application_train.csv
```

The dashboard and training pipeline are designed around `application_train.csv`. Additional Home Credit tables can be incorporated later for a richer feature store.

## Local setup

```bash
git clone <your-repository-url>
cd credit-risk-scoring

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Put `application_train.csv` in `data/raw/`.

## Train

```bash
python -m src.models.train \
  --data data/raw/application_train.csv \
  --output models
```

This produces:

```text
models/
├── logistic_regression.joblib
├── xgboost.joblib
└── metrics.json
```

MLflow runs are stored under `mlruns/`.

## MLflow

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open http://localhost:5000.

For a Dockerized setup:

```bash
docker compose up mlflow
```

## Dashboard

After training:

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501.

## Tests

```bash
pytest -q
```

## Lint

```bash
ruff check src dashboard tests
```

## Docker

```bash
docker compose up --build
```

Dashboard: http://localhost:8501

MLflow: http://localhost:5000

## Project structure

```text
src/
├── data/
├── features/
├── models/
├── segmentation/
└── profitability/

dashboard/
tests/
notebooks/
configs/
models/
data/
.github/workflows/
```

## Modeling notes

The target is the Home Credit `TARGET` column, where `1` indicates default and `0` indicates no default.

The training pipeline uses:

- stratified train/validation split
- class weighting for Logistic Regression
- `scale_pos_weight` for XGBoost
- median imputation for numerical features
- most-frequent imputation + one-hot encoding for categorical features

The KS statistic is calculated as the maximum separation between the empirical cumulative distributions of positive and negative observations.

## Profitability assumptions

Profitability is illustrative rather than a production lending policy. The default configuration assumes:

- LGD = 45%
- revenue = 12% of exposure
- operating cost = €100

These assumptions should be replaced with institution-specific estimates before real-world use.

## Reproducibility

Random seed: `42`.

The project intentionally does **not** commit the Kaggle dataset, trained models, or MLflow artifacts. This keeps the repository lightweight and avoids redistributing the dataset.

## Future improvements

- Add Home Credit bureau/balance/installment/credit-card tables
- Optuna hyperparameter optimization
- Probability calibration
- SHAP explanations
- Population Stability Index (PSI)
- model monitoring
- fairness analysis
- feature-store layer
- deployment to a cloud service
