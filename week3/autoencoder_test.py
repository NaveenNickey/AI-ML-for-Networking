# autoencoder_test.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# === Step 1: Load Mixed Data (Benign + Attack) ===
file_path = "/home/kali/Downloads/Test_CIC.csv"  # Change to test data
df_test = pd.read_csv(file_path)

# === Step 2: Save True Labels for Evaluation ===
true_labels = df_test['Label']
y_true = np.where(true_labels == 'BENIGN', 0, 1)  # 0 = benign, 1 = attack

# === Step 3: Preprocess ===
df_test = df_test.select_dtypes(include=[np.number])
df_test.fillna(0, inplace=True)

# === Step 4: Load Scaler and Model ===
scaler = joblib.load("autoencoder_scaler.joblib")
autoencoder = load_model("autoencoder_model.h5")

X_test = scaler.transform(df_test)

# === Step 5: Predict and Compute Errors ===
X_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_pred, 2), axis=1)

# === Step 6: Choose Threshold ===
threshold = np.percentile(mse, 95)  # Top 5% = anomaly
predictions = (mse > threshold).astype(int)  # 1 = anomaly, 0 = normal

# === Step 7: Evaluate ===
print("✅ Anomaly Detection Results")
print("Reconstruction Threshold:", threshold)
print(classification_report(y_true, predictions))

# === Step 8: Plot Reconstruction Error Distribution ===
plt.figure(figsize=(10,5))
plt.hist(mse, bins=50)
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.show()
