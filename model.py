# =========================================================
# MODEL 1 — Predictive Maintenance (REAL DATA VERSION)
# Uses: sensor_data.csv + failures.csv
# Output: model1_failure_risk_REAL.pkl
# =========================================================

import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
from imblearn.over_sampling import SMOTE

# =========================================================
# STEP 1 — LOAD SENSOR DATA
# =========================================================

print("✔ Loading sensor_data.csv ...")
df = pd.read_csv("sensor_data.csv", parse_dates=["date"])
df.sort_values(["rake_id", "date"], inplace=True)
df.reset_index(drop=True, inplace=True)
print("✔ Rows loaded:", df.shape)

# =========================================================
# STEP 2 — LOAD FAILURE LOG & CREATE LABELS
# =========================================================

print("✔ Loading failures.csv ...")
fail = pd.read_csv("failures.csv", parse_dates=["failure_date"])

# Group failures by rake
fail_map = (
    fail.groupby("rake_id")["failure_date"]
        .apply(list)
        .to_dict()
)

# Initialize label
df["failure_next_7_days"] = 0

print("✔ Generating labels...")

for i in range(len(df)):
    rake = df.loc[i, "rake_id"]
    date = df.loc[i, "date"]

    if rake not in fail_map:
        continue

    # check if any failure within next 7 days
    failures = fail_map[rake]
    if any(0 < (f - date).days <= 7 for f in failures):
        df.loc[i, "failure_next_7_days"] = 1

print("✔ Total positive samples (label=1):", df["failure_next_7_days"].sum())

# =========================================================
# STEP 3 — FEATURE ENGINEERING (ROLLING WINDOWS)
# =========================================================

print("✔ Computing rolling window features...")

df["rolling_7d_km"] = (
    df.groupby("rake_id")["daily_km"].rolling(7).sum().reset_index(0, drop=True)
)

df["rolling_7d_hvac"] = (
    df.groupby("rake_id")["hvac_faults"].rolling(7).sum().reset_index(0, drop=True)
)

df["rolling_7d_brake_temp"] = (
    df.groupby("rake_id")["brake_temp_avg"].rolling(7).mean().reset_index(0, drop=True)
)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("✔ Final dataset shape:", df.shape)

# =========================================================
# STEP 4 — TRAIN/TEST SPLIT (TIME-BASED)
# =========================================================

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

print("✔ Train:", X_train.shape, " Test:", X_test.shape)
print("✔ Train class distribution:\n", y_train.value_counts())

# =========================================================
# STEP 5 — BALANCE USING SMOTE
# =========================================================

print("✔ Applying SMOTE balancing...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("✔ After SMOTE class balance:\n", y_train_sm.value_counts())

# =========================================================
# STEP 6 — TRAIN MODEL (RandomForest)
# =========================================================

print("✔ Training RandomForest model...")

model = RandomForestClassifier(
    n_estimators=600,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train_sm, y_train_sm)

# =========================================================
# STEP 7 — EVALUATE MODEL
# =========================================================

print("\n================ DEFAULT THRESHOLD REPORT ================")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print("✔ Accuracy:", accuracy_score(y_test, y_pred))
print("✔ ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Probability threshold tuning
print("\n================ CUSTOM THRESHOLD (0.5) ================")

y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.50
y_pred_custom = (y_prob >= threshold).astype(int)

print(classification_report(y_test, y_pred_custom))
print(confusion_matrix(y_test, y_pred_custom))

print("✔ Custom Threshold ROC-AUC:", roc_auc_score(y_test, y_prob))

# =========================================================
# STEP 8 — SAVE MODEL + FEATURE IMPORTANCE
# =========================================================

joblib.dump(model, "model1_failure_risk_REAL.pkl")
print("\n✔ Model saved as: model1_failure_risk_REAL.pkl")

# Save feature importances
fi = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

fi.to_csv("feature_importance.csv", index=False)
print("✔ Feature importance saved → feature_importance.csv")
