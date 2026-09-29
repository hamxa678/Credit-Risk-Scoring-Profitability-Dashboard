import os
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.features.build_features import add_domain_features, make_customer_features
from src.models.predict import risk_band, risk_score
from src.profitability.expected_profit import expected_loss, expected_profit

st.set_page_config(page_title="Credit Risk & Profitability", page_icon="💳", layout="wide")

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost.joblib")
DATA_PATH = os.getenv("DATA_PATH", "data/raw/application_train.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    path = Path(DATA_PATH)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

st.title("💳 Credit Risk Scoring & Profitability Dashboard")
st.caption("Home Credit default-risk model • XGBoost • customer segmentation • expected profitability")

df = load_data()
if df.empty:
    st.warning("Place Home Credit's application_train.csv in data/raw/ and restart the app.")
    st.stop()

model = load_model()

st.sidebar.header("Customer")
if "SK_ID_CURR" in df.columns:
    customer_id = st.sidebar.selectbox("Customer ID", df["SK_ID_CURR"].head(5000).tolist())
    row = df[df["SK_ID_CURR"] == customer_id].iloc[[0]]
else:
    row = df.iloc[[0]]
    customer_id = 0

customer = add_domain_features(row.copy())
pd_value = float(model.predict_proba(customer)[:, 1][0])

exposure = float(row["AMT_CREDIT"].iloc[0]) if "AMT_CREDIT" in row else 10000.0
income = float(row["AMT_INCOME_TOTAL"].iloc[0]) if "AMT_INCOME_TOTAL" in row else 30000.0
revenue = 0.12 * exposure
loss = expected_loss(pd_value, exposure)
profit = expected_profit(pd_value, exposure, revenue)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Probability of Default", f"{pd_value:.1%}")
c2.metric("Risk Score", risk_score(pd_value))
c3.metric("Expected Loss", f"€{loss:,.0f}")
c4.metric("Expected Profit", f"€{profit:,.0f}")

st.subheader("Risk profile")
st.write(f"**Risk band:** {risk_band(pd_value)}")
st.write(f"**Credit exposure:** €{exposure:,.0f}  •  **Income:** €{income:,.0f}")

if "TARGET" in df.columns:
    st.subheader("Portfolio default distribution")
    fig = px.histogram(df, x="TARGET", title="Observed target distribution")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Customer record")
st.dataframe(row.T.rename(columns={row.index[0]: "Value"}), use_container_width=True)
