# ===============================================
# STREAMLIT APP — Predictive Maintenance (Model-1)
# FIXED VERSION WITH DEBUG + MODEL CHECKS
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import sklearn

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(
    page_title="KMRL - Predictive Maintenance (Model-1)",
    layout="wide"
)

# ----------------------------
# Model Path + Feature Columns
# ----------------------------
MODEL_FILE = os.getenv("MODEL_PATH", "model1_failure_risk_REAL.pkl")
USE_FLASK = False

FEATURE_COLUMNS = [
    "daily_km",
    "cumulative_km",
    "brake_temp_avg",
    "hvac_faults",
    "door_faults",
    "rolling_7d_km",
    "rolling_7d_hvac",
    "rolling_7d_brake_temp",
]

FLASK_URL = os.getenv("FLASK_URL", "http://127.0.0.1:5000/predict_from_features")

# ----------------------------
# Debug Information (VERY IMPORTANT)
# ----------------------------
st.sidebar.subheader("🔍 Debug Info")
st.sidebar.write("📁 MODEL_PATH:", MODEL_FILE)
st.sidebar.write("📦 Model file exists:", os.path.exists(MODEL_FILE))
st.sidebar.write("⚙️ sklearn version:", sklearn.__version__)

# ----------------------------
# Load Model Function
# ----------------------------
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {path}")
        return None
    try:
        model = joblib.load(path)
        st.sidebar.success("✔ Model loaded successfully")
        return model
    except Exception as e:
        st.sidebar.error("❌ Failed to load model")
        st.sidebar.write(e)
        return None

# ----------------------------
# Prediction Functions
# ----------------------------
def predict_with_model(model, feature_dict):
    X = np.array([[feature_dict[c] for c in FEATURE_COLUMNS]])
    prob = float(model.predict_proba(X)[0][1])
    return prob

def risk_from_prob(p, medium=0.3, high=0.7):
    if p >= high:
        return "HIGH"
    elif p >= medium:
        return "MEDIUM"
    else:
        return "LOW"

# ----------------------------
# Load model once
# ----------------------------
model = load_model(MODEL_FILE)

# ----------------------------
# UI Layout
# ----------------------------
st.title("🚇 KMRL — Predictive Maintenance (Model-1)")
st.caption("Predict probability of equipment failure for Kochi Metro rakes.")

mode = st.sidebar.radio("Choose Mode:", [
    "Single input",
    "Batch CSV",
    "Model info",
    "What-if simulator"
])

# ----------------------------
# SINGLE INPUT MODE
# ----------------------------
if mode == "Single input":
    st.header("🔹 Single Prediction")

    col1, col2 = st.columns(2)

    with col1:
        daily_km = st.number_input("daily_km", 50.0, 300.0, 180.0)
        cumulative_km = st.number_input("cumulative_km", 10000.0, 300000.0, 90000.0)
        brake_temp_avg = st.number_input("brake_temp_avg", 40.0, 200.0, 120.0)
        hvac_faults = st.number_input("hvac_faults", 0, 20, 5)
    with col2:
        door_faults = st.number_input("door_faults", 0, 10, 1)
        rolling_7d_km = st.number_input("rolling_7d_km", 500.0, 3000.0, 1300.0)
        rolling_7d_hvac = st.number_input("rolling_7d_hvac", 0, 40, 12)
        rolling_7d_brake_temp = st.number_input("rolling_7d_brake_temp", 40.0, 200.0, 95.0)

    if st.button("Predict Failure Risk"):
        features = {
            "daily_km": daily_km,
            "cumulative_km": cumulative_km,
            "brake_temp_avg": brake_temp_avg,
            "hvac_faults": hvac_faults,
            "door_faults": door_faults,
            "rolling_7d_km": rolling_7d_km,
            "rolling_7d_hvac": rolling_7d_hvac,
            "rolling_7d_brake_temp": rolling_7d_brake_temp,
        }

        if model is None:
            st.error("❌ Model not loaded.")
        else:
            prob = predict_with_model(model, features)
            risk = risk_from_prob(prob)

            st.metric("Failure Probability", f"{prob:.3f}")
            st.success(f"Risk Level: **{risk}**")

            st.json(features)

# ----------------------------
# BATCH CSV MODE
# ----------------------------
elif mode == "Batch CSV":
    st.header("📂 Batch CSV Prediction")
    st.write("Upload CSV containing model feature columns:")
    st.code(", ".join(FEATURE_COLUMNS))

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            results = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                prob = predict_with_model(model, row_dict)
                risk = risk_from_prob(prob)
                results.append({**row_dict, "failure_probability": prob, "risk_level": risk})

            out_df = pd.DataFrame(results)
            st.dataframe(out_df)

# ----------------------------
# MODEL INFORMATION
# ----------------------------
elif mode == "Model info":
    st.header("📊 Model Info")
    if model is not None:
        st.write(model)
        try:
            fi_df = pd.DataFrame({
                "feature": FEATURE_COLUMNS,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)

            st.table(fi_df)
            st.bar_chart(fi_df.set_index("feature"))
        except:
            st.warning("Feature importance unavailable.")

# ----------------------------
# WHAT-IF SIMULATOR
# ----------------------------
elif mode == "What-if simulator":
    st.header("🔍 What-if Simulator")
    st.write("Adjust values to see effect on failure probability")

    base = {c: 0 for c in FEATURE_COLUMNS}
    base["daily_km"] = st.slider("daily_km", 50, 300, 180)
    base["cumulative_km"] = st.slider("cumulative_km", 50000, 200000, 90000)
    base["brake_temp_avg"] = st.slider("brake_temp_avg", 40, 200, 120)
    base["hvac_faults"] = st.slider("hvac_faults", 0, 20, 5)
    base["door_faults"] = st.slider("door_faults", 0, 10, 1)
    base["rolling_7d_km"] = st.slider("rolling_7d_km", 500, 3000, 1300)
    base["rolling_7d_hvac"] = st.slider("rolling_7d_hvac", 0, 40, 12)
    base["rolling_7d_brake_temp"] = st.slider("rolling_7d_brake_temp", 50, 200, 95)

    if st.button("Simulate"):
        prob = predict_with_model(model, base)
        risk = risk_from_prob(prob)
        st.metric("Failure Probability", f"{prob:.3f}")
        st.success(f"Risk Level: **{risk}**")
        st.json(base)
