import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("mohs_model.joblib")  

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model, feats = bundle["model"], bundle["features"]

model, feats = bundle["model"], bundle["features"]

st.title("Mohs Hardness Predictor")


with st.form("predict"):
    inputs = {f: st.number_input(f, value=0.0, format="%.6f") for f in feats}
    submitted = st.form_submit_button("🔮 Predict")


if submitted:
    X = pd.DataFrame([inputs], columns=feats)
    y_hat = model.predict(X)[0]
    st.metric("Predicted Hardness", f"{y_hat:.4f}")
