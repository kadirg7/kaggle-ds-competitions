from pathlib import Path
from joblib import load
import numpy as np
import pandas as pd
import streamlit as st

# MODEL DOSYA YOLU (iki olasılığı da dener)
ROOT = Path(__file__).resolve().parent
CANDIDATES = [
    ROOT / "abalone_xgb_artifacts.joblib",      # kökte ise
    ROOT / "src" / "abalone_xgb_artifacts.joblib"  # src/ içinde ise
]
ART_PATH = next(p for p in CANDIDATES if p.exists())

ART = load(ART_PATH)
model, scaler   = ART["model"], ART["scaler"]
F_COLS          = ART["feature_cols"]
NUM_COLS        = ART["numeric_cols"]
SEX_LEVELS      = ART.get("sex_levels", ["F","I","M"])


st.set_page_config(page_title="Abalone Rings", page_icon="🐚")
st.title("🐚 Abalone Rings – Single Prediction")


c = st.columns(3)
sex       = st.selectbox("Sex", SEX_LEVELS)
length    = c[0].number_input("Length",   0.0, 1.2, 0.55, 0.005)
diameter  = c[1].number_input("Diameter", 0.0, 1.0, 0.43, 0.005)
height    = c[2].number_input("Height",   0.0, 0.4,  0.15, 0.005)
whole_w   = c[0].number_input("Whole weight",   0.0, 3.0, 0.78, 0.005)
whole_w1  = c[1].number_input("Whole weight.1", 0.0, 1.8, 0.33, 0.005)
whole_w2  = c[2].number_input("Whole weight.2", 0.0, 0.8, 0.16, 0.005)
shell_w   = st.number_input("Shell weight", 0.0, 1.1, 0.24, 0.005)


def to_features(raw: pd.DataFrame) -> pd.DataFrame:
    
    for s in SEX_LEVELS:
        raw[f"Sex_{s}"] = (raw["Sex"] == s).astype(int)
    raw = raw.drop(columns="Sex")

    for col in F_COLS:
        if col not in raw:
            raw[col] = 0
    raw[NUM_COLS] = scaler.transform(raw[NUM_COLS])
    return raw[F_COLS]

if st.button("Predict"):
    raw = pd.DataFrame([{
        "Sex": sex, "Length": length, "Diameter": diameter, "Height": height,
        "Whole weight": whole_w, "Whole weight.1": whole_w1,
        "Whole weight.2": whole_w2, "Shell weight": shell_w
    }])
    X = to_features(raw)
    y = model.predict(X)
    rings = float(np.maximum(np.expm1(y)[0], 0))  
    st.success(f"Rings: **{rings:.2f}**  |  Age ≈ **{rings + 1.5:.2f}** years")

st.caption("Model & scaler loaded from abalone_xgb_artifacts.joblib")
