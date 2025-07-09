import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Load Dataset ===
file_path = "/home/kali/Downloads/facebook.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # Strip column names

print("✅ Loaded columns:", df.columns.tolist())

# === Step 2: Keep Numeric Columns Only ===
numeric_df = df.select_dtypes(include=[np.number])
print("✅ Numeric columns used for anomaly detection:", numeric_df.columns.tolist())

if numeric_df.empty:
    raise ValueError("❌ No numeric columns found in the dataset. Cannot continue.")

# === Step 3: Replace NaNs and Infs ===
X = numeric_df.replace([np.inf, -np.inf], np.nan)
X.fillna(0, inplace=True)

# === Step 4: Normalize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 5: Train Isolation Forest ===
print("\n🚀 Training Isolation Forest for anomaly detection...")
clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
clf.fit(X_scaled)

# === Step 6: Predict Anomalies ===
y_pred = clf.predict(X_scaled)  # -1 = anomaly, 1 = normal
df['Anomaly'] = y_pred

# === Step 7: Visualize ===
plt.figure(figsize=(10, 5))
sns.countplot(x='Anomaly', data=df)
plt.title("Anomaly vs Normal Count (Unsupervised Detection)")
plt.xlabel("Anomaly Label (-1 = Anomaly, 1 = Normal)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# === Step 8: Save Detected Anomalies ===
anomalies = df[df['Anomaly'] == -1]
anomalies.to_csv("detected_anomalies.csv", index=False)
print(f"\n✅ {len(anomalies)} anomalies saved to 'detected_anomalies.csv'")

