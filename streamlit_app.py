import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ✅ Full width layout
st.set_page_config(page_title="Stroke Monitoring", layout="wide")

# === Main Title ===
st.markdown(
    "<h1 style='text-align: center; color: white;'>🩺 Stroke Prediction Monitoring Dashboard</h1>",
    unsafe_allow_html=True
)

log_file = "logs/prediction_logs.csv"
if not os.path.exists(log_file):
    st.warning("⚠️ No prediction data available yet.")
    st.stop()

# === Load prediction logs ===
df = pd.read_csv(log_file, parse_dates=["timestamp"])

# === General Statistics ===
st.markdown("### 📊 General Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", len(df))
col2.metric("Stroke Predictions (1)", df["prediction"].sum())
col3.metric("Stroke Percentage", f"{(df['prediction'].mean() * 100):.2f}%")

st.markdown("---")

# === Bar Chart: Stroke vs Non-Stroke ===
st.markdown("### 🧠 Prediction Distribution")
count_df = df["prediction"].value_counts().rename({0: "No Stroke", 1: "Stroke"})
st.bar_chart(count_df)

st.markdown("---")

# === Daily Prediction Trend Chart ===
st.markdown("### 📅 Daily Prediction Trend")
df["date"] = df["timestamp"].dt.date
daily = df.groupby(["date", "prediction"]).size().unstack(fill_value=0)
st.line_chart(daily)

st.markdown("---")

# === Additional Statistics ===
st.markdown("### 🧬 Average Age & Glucose Level by Prediction Category")
summary = df.groupby("prediction")[["age", "avg_glucose_level"]].mean().rename(index={0: "No Stroke", 1: "Stroke"})
st.dataframe(summary.style.format("{:.2f}"))

# === Footer ===
st.markdown("---")
st.caption("📍 AI Model Monitoring • Powered by Streamlit 🚀")
