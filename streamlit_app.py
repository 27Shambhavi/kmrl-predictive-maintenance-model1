# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

st.set_page_config(page_title="KMRL - Predictive Maintenance (Model-1)", layout="wide")
MODEL_FILE = os.getenv("MODEL_PATH", "model1_failure_risk_REAL.pkl")
USE_FLASK = False  # if True, Streamlit will call the Flask API instead of loading the model

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

# ---------- Helper functions ----------
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        return None
    model = joblib.load(path)
    return model

def predict_with_model(model, feature_dict):
    x = np.array([[feature_dict[c] for c in FEATURE_COLUMNS]])
    prob = float(model.predict_proba(x)[0][1])
    return prob

def predict_via_flask(payload):
    resp = requests.post(FLASK_URL, json=payload, timeout=10)
    return resp.json()

def risk_from_prob(p, med=0.3, high=0.7):
    if p >= high:
        return "HIGH"
    elif p >= med:
        return "MEDIUM"
    else:
        return "LOW"

# ---------- Sidebar ----------
st.sidebar.title("KMRL — Model-1")
st.sidebar.markdown("**Predictive Maintenance — Failure Risk**")
mode = st.sidebar.radio("Mode", ["Single input", "Batch CSV", "Model info", "What-if simulator"])

st.sidebar.markdown("---")
st.sidebar.write("Model file:", MODEL_FILE)
if not USE_FLASK:
    st.sidebar.write("Running local model")
else:
    st.sidebar.write("Calling Flask API at:")
    st.sidebar.write(FLASK_URL)

# ---------- Load model (if local) ----------
model = None
if not USE_FLASK:
    model = load_model(MODEL_FILE)

# ---------- Main UI ----------
st.title("KMRL — Predictive Maintenance (Model-1)")
st.caption("Predict failure probability for each rake. Input fields must match model features.")

if mode == "Single input":
    st.subheader("Single prediction")
    col1, col2 = st.columns(2)

    with col1:
        daily_km = st.number_input("daily_km", value=180.0)
        cumulative_km = st.number_input("cumulative_km", value=90500.0)
        brake_temp_avg = st.number_input("brake_temp_avg", value=118.5)
        hvac_faults = st.number_input("hvac_faults", value=5, step=1)
    with col2:
        door_faults = st.number_input("door_faults", value=1, step=1)
        rolling_7d_km = st.number_input("rolling_7d_km", value=1280.0)
        rolling_7d_hvac = st.number_input("rolling_7d_hvac", value=12, step=1)
        rolling_7d_brake_temp = st.number_input("rolling_7d_brake_temp", value=97.2)

    if st.button("Predict"):
        input_features = {
            "daily_km": float(daily_km),
            "cumulative_km": float(cumulative_km),
            "brake_temp_avg": float(brake_temp_avg),
            "hvac_faults": int(hvac_faults),
            "door_faults": int(door_faults),
            "rolling_7d_km": float(rolling_7d_km),
            "rolling_7d_hvac": int(rolling_7d_hvac),
            "rolling_7d_brake_temp": float(rolling_7d_brake_temp),
        }

        try:
            if USE_FLASK:
                resp = predict_via_flask(input_features)
                prediction = resp.get("prediction", {})
                prob = prediction.get("failure_probability", None)
                risk = prediction.get("risk_level", None)
            else:
                if model is None:
                    st.error("Model not loaded.")
                    st.stop()
                prob = predict_with_model(model, input_features)
                risk = risk_from_prob(prob)

            st.metric("Failure probability", f"{prob:.3f}")
            st.info(f"Risk level: **{risk}**")

            st.subheader("Input features")
            st.json(input_features)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif mode == "Batch CSV":
    st.subheader("Batch prediction from CSV")
    st.markdown("Upload a CSV with columns: " + ", ".join(FEATURE_COLUMNS))
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        if st.button("Run batch prediction"):
            results = []
            for i, row in df.iterrows():
                payload = {c: row[c] for c in FEATURE_COLUMNS}
                try:
                    if USE_FLASK:
                        resp = predict_via_flask(payload)
                        out = resp.get("prediction", {})
                        prob = out.get("failure_probability", None)
                        risk = out.get("risk_level", None)
                    else:
                        prob = predict_with_model(model, payload)
                        risk = risk_from_prob(prob)
                    results.append({**payload, "failure_probability": prob, "risk_level": risk})
                except Exception as e:
                    results.append({**payload, "error": str(e)})

            out_df = pd.DataFrame(results)
            st.success("Batch done")
            st.dataframe(out_df)
            csv = out_df.to_csv(index=False)
            st.download_button("Download results CSV", csv, file_name="predictions_results.csv", mime="text/csv")

elif mode == "Model info":
    st.subheader("Model information & feature importance")
    if not USE_FLASK and model is not None:
        st.write(model)
        try:
            fi = model.feature_importances_
            fi_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": fi}).sort_values("importance", ascending=False)
            st.bar_chart(fi_df.set_index("feature"))
            st.table(fi_df)
        except Exception as e:
            st.write("Feature importance not available:", e)
    else:
        st.info("Model not loaded locally. Use Flask mode or place model file in project folder.")

elif mode == "What-if simulator":
    st.subheader("What-if simulation (single rake)")
    st.markdown("Adjust feature sliders and see how probability changes.")
    base = {
        "daily_km": st.slider("daily_km", 100, 260, 180),
        "cumulative_km": st.slider("cumulative_km", 80000, 150000, 90500),
        "brake_temp_avg": st.slider("brake_temp_avg", 40, 160, 118),
        "hvac_faults": st.slider("hvac_faults", 0, 12, 5),
        "door_faults": st.slider("door_faults", 0, 5, 1),
        "rolling_7d_km": st.slider("rolling_7d_km", 700, 2200, 1280),
        "rolling_7d_hvac": st.slider("rolling_7d_hvac", 0, 30, 12),
        "rolling_7d_brake_temp": st.slider("rolling_7d_brake_temp", 40, 150, 97),
    }
    st.write(base)
    if st.button("Simulate"):
        if USE_FLASK:
            resp = predict_via_flask(base)
            pred = resp.get("prediction", {})
            st.metric("Failure prob", pred.get("failure_probability"))
            st.info("Risk: " + str(pred.get("risk_level")))
        else:
            prob = predict_with_model(model, base)
            st.metric("Failure prob", f"{prob:.3f}")
            st.info("Risk: " + risk_from_prob(prob))