import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# === Load x_test and y_meta ===
print("🚀 Loading meta features and labels...")

X_raw = pd.read_csv("x_test.csv")  # Should contain traffic_pred, threat_pred, anomaly_flag
y_test = pd.read_csv("y_meta.csv", header=None).squeeze()

print("✅ Files loaded successfully.")
print("📊 X_raw shape:", X_raw.shape)
print("🔖 Sample y_test labels:", y_test.unique()[:10])

# === Auto-truncate y_test if longer ===
if len(y_test) > len(X_raw):
    y_test = y_test.iloc[:len(X_raw)]
    print(f"✂️ Truncated y_test to match X_raw length: {len(y_test)}")

# === Encode string labels in X_raw (meta features)
label_encoders = {}
for col in ["traffic_pred", "threat_pred"]:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col])
    label_encoders[col] = le

# Convert anomaly_flag to integer if not already
X_raw["anomaly_flag"] = pd.to_numeric(X_raw["anomaly_flag"], errors="coerce")

# === Encode y_test labels
y_le = LabelEncoder()
y_encoded = y_le.fit_transform(y_test)

# === Train meta-model
print("🧠 Training meta-model (Logistic Regression)...")
meta_model = LogisticRegression(max_iter=500)
meta_model.fit(X_raw, y_encoded)

# === Predict and evaluate
y_pred = meta_model.predict(X_raw)

print("📈 Confusion Matrix:")
print(confusion_matrix(y_encoded, y_pred))

print("\n📋 Classification Report:")
print(classification_report(y_encoded, y_pred, target_names=y_le.classes_))

# Optional: Save the model
joblib.dump(meta_model, "stacked_meta_model.pkl")
print("💾 Saved: stacked_meta_model.pkl")

