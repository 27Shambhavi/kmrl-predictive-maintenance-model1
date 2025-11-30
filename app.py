# ============================================================
# app.py (REAL DATA VERSION for Model-1 Predictive Maintenance)
# Loads model1_failure_risk_REAL.pkl
# ============================================================

import os
import sys
import joblib
import traceback
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# Feature columns for REAL DATA model
# -------------------------------------------------------------
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

# Risk thresholds
HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.30

# -------------------------------------------------------------
# Startup logs
# -------------------------------------------------------------
print("Starting REAL Model-1 API ...", file=sys.stderr)

# Model file path
MODEL_PATH = os.getenv("MODEL_PATH", "model1_failure_risk_REAL.pkl")
print("Resolved MODEL_PATH:", MODEL_PATH, file=sys.stderr)

_model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)
    print("MODEL LOADED SUCCESSFULLY!", file=sys.stderr)
except Exception as e:
    print("ERROR: Model failed to load", str(e), file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

# -------------------------------------------------------------
# Flask App Init
# -------------------------------------------------------------
app = Flask(__name__)

def _model_available():
    return _model is not None

def _predict_from_row(feature_dict):
    if not _model_available():
        raise RuntimeError("Model not loaded")

    # Convert features → numpy array
    x = np.array([[feature_dict[col] for col in FEATURE_COLUMNS]])
    prob_failure = float(_model.predict_proba(x)[0][1])

    # Assign risk class
    if prob_failure >= HIGH_THRESHOLD:
        risk = "HIGH"
    elif prob_failure >= MEDIUM_THRESHOLD:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "failure_probability": round(prob_failure, 3),
        "risk_level": risk
    }

# -------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "model_loaded": _model_available(),
        "model_path": MODEL_PATH
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "KMRL Model-1 Predictive Maintenance API Running!",
        "model_loaded": _model_available()
    })

# -------------------------------------------------------------
# Predict (Single JSON Row)
# -------------------------------------------------------------
@app.route("/predict_from_features", methods=["POST"])
def predict_from_features():
    try:
        if not _model_available():
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        # Check missing fields
        missing = [f for f in FEATURE_COLUMNS if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Convert all to float
        try:
            feature_dict = {f: float(data[f]) for f in FEATURE_COLUMNS}
        except:
            return jsonify({"error": "All fields must be numeric"}), 400

        # Predict
        result = _predict_from_row(feature_dict)

        return jsonify({
            "input_features": feature_dict,
            "prediction": result
        })

    except Exception as exc:
        print("Error:", str(exc), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify({"error": "Server Error", "details": str(exc)}), 500

# -------------------------------------------------------------
# Predict from CSV
# -------------------------------------------------------------
@app.route("/predict_from_csv", methods=["POST"])
def predict_from_csv():
    try:
        if not _model_available():
            return jsonify({"error": "Model not loaded"}), 500

        if "file" not in request.files:
            return jsonify({"error": "Upload CSV as form-data named 'file'"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        results = []
        for i, row in df.iterrows():
            try:
                feature_dict = {c: float(row[c]) for c in FEATURE_COLUMNS}
                pred = _predict_from_row(feature_dict)
                results.append({"index": int(i), "input": feature_dict, "output": pred})
            except Exception as e:
                results.append({"index": int(i), "error": str(e)})

        return jsonify({
            "total_rows": len(results),
            "results": results
        })

    except Exception as exc:
        print("CSV Prediction Error:", str(exc), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify({"error": "Server Error", "details": str(exc)}), 500

# -------------------------------------------------------------
# Run Server
# -------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("1", "true", "yes")

    print(f"Running on http://{host}:{port}", file=sys.stderr)
    app.run(host=host, port=port, debug=debug)

