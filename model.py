
# MODEL 1 — Predictive Maintenance (REAL DATA VERSION)
# Using: sensor_data.csv + failures.csv


import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ============================================
# STEP 1 — LOAD SENSOR DATA
# ============================================

print("Loading sensor_data.csv ...")
df = pd.read_csv("sensor_data.csv", parse_dates=["date"])
df = df.sort_values(["rake_id", "date"]).reset_index(drop=True)

print("Rows loaded:", df.shape)

# ============================================
# STEP 2 — LOAD FAILURE LOG & MERGE TO CREATE LABEL
# ============================================

print("Loading failures.csv ...")
fail = pd.read_csv("failures.csv", parse_dates=["failure_date"])

# failure events per rake
failures_by_rake = (
    fail.groupby("rake_id")["failure_date"]
    .apply(list)
    .to_dict()
)

# create binary label: failure in next 7 days
df["failure_next_7_days"] = 0

print("Generating labels (may take ~10 seconds)...")

for idx in range(len(df)):
    rake = df.loc[idx, "rake_id"]
    date = df.loc[idx, "date"]

    if rake not in failures_by_rake:
        continue

    # check if any failure occurs within next 7 days
    for fdate in failures_by_rake[rake]:
        if 0 < (fdate - date).days <= 7:
            df.loc[idx, "failure_next_7_days"] = 1
            break

print("Total failures (label=1):", df["failure_next_7_days"].sum())

# ============================================
# STEP 3 — FEATURE ENGINEERING (ROLLING WINDOWS)
# ============================================

print("Computing rolling features...")

df["rolling_7d_km"] = (
    df.groupby("rake_id")["daily_km"]
      .rolling(7).sum().reset_index(0, drop=True)
)

df["rolling_7d_hvac"] = (
    df.groupby("rake_id")["hvac_faults"]
      .rolling(7).sum().reset_index(0, drop=True)
)

df["rolling_7d_brake_temp"] = (
    df.groupby("rake_id")["brake_temp_avg"]
      .rolling(7).mean().reset_index(0, drop=True)
)

df = df.dropna().reset_index(drop=True)
print("Final dataset shape:", df.shape)

# ============================================
# STEP 4 — TRAIN/TEST SPLIT (TIME-BASED)
# ============================================

features = [
    "daily_km", "cumulative_km",
    "brake_temp_avg", "hvac_faults", "door_faults",
    "rolling_7d_km", "rolling_7d_hvac", "rolling_7d_brake_temp"
]

X = df[features]
y = df["failure_next_7_days"]

split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("Train:", X_train.shape, "Test:", X_test.shape)

# ============================================
# STEP 5 — SMOTE FOR CLASS BALANCING
# ============================================

print("Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("After SMOTE:", y_train_sm.value_counts())

# ============================================
# STEP 6 — TRAIN RANDOM FOREST MODEL
# ============================================

print("Training RandomForest...")
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=16,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_sm, y_train_sm)

# ============================================
# STEP 7 — EVALUATE MODEL
# ============================================

y_pred = model.predict(X_test)

print("\n===== REPORT (DEFAULT THRESHOLD) =====")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# probability threshold tuning
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.30
y_pred_custom = (y_prob >= threshold).astype(int)

print("\n===== REPORT (Custom threshold = 0.30) =====")
print(classification_report(y_test, y_pred_custom))
print(confusion_matrix(y_test, y_pred_custom))

# ============================================
# STEP 8 — SAVE MODEL
# ============================================

joblib.dump(model, "model1_failure_risk_REAL.pkl")
print("Model saved as: model1_failure_risk_REAL.pkl")

