import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.set_page_config(page_title="ML Monitoring Dashboard")

st.title("Credit Risk ML Monitoring System")
st.markdown("---")

with open("monitoring/monitoring_metrics.json") as f:
    metrics = json.load(f)

st.subheader("Data Drift Status")

col1, col2 = st.columns(2)

col1.metric(
    "Dataset Drift",
    "Detected" if metrics["data_drift"] else "Stable"
)

col2.metric(
    "Drifted Features %",
    f"{metrics['drifted_features_pct']:.2f}%"
)

st.subheader("Model Performance")

col1, col2 = st.columns(2)

col1.metric(
    "ROC-AUC",
    f"{metrics['roc_auc']:.3f}"
)

col2.metric(
    "Recall",
    f"{metrics['recall']:.3f}"
)
df = pd.read_csv("logs/predictions.csv")
st.subheader("Prediction Risk Distribution")

fig = px.histogram(
    df,
    x="prediction",
    nbins=50,
    title="Default Probability Distribution"
)

st.plotly_chart(fig)


retrain_log = pd.read_csv(
    "retraining/retraining_log.csv"
)

st.subheader("Retraining History")

st.dataframe(retrain_log)

with open("models/registry.json") as f:
    registry = json.load(f)

st.subheader("Production Model")

st.info(
    f"Active Model: {registry['production_model']}"
)
