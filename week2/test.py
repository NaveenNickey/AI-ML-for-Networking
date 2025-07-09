import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")

# === Step 1: Load Dataset ===
print("📥 Loading dataset...")
df = pd.read_csv("/home/kali/Downloads/CIC-IDS_2018.csv")

# === Step 2: Clean Data ===
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# === Step 3: Normalize column names ===
df.columns = df.columns.str.strip()

# === Step 4: Keep only BENIGN samples ===
df["Label"] = df["Label"].astype(str).str.strip()
df = df[df["Label"] == "Benign"]
print("✅ BENIGN rows found:", df.shape[0])

if df.empty:
    raise ValueError("❌ No BENIGN rows after filtering. Check your dataset.")

# === Step 5: Drop label and non-numeric columns ===
df.drop(columns=["Label", "Timestamp"], inplace=True, errors="ignore")
df = df.select_dtypes(include=[np.number])

# === Step 6: Scale Features ===
print("⚙️ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# === Step 7: Train Isolation Forest ===
print("🤖 Training Isolation Forest model...")
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_scaled)

# === Step 8: Save Model & Scaler ===
joblib.dump(model, "anomaly_model.pkl")
joblib.dump(scaler, "anomaly_scaler.pkl")

print("🎉 Saved: anomaly_model.pkl and anomaly_scaler.pkl")

