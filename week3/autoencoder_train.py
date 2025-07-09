# autoencoder_train.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === Step 1: Load Normal-Only Data ===
file_path = "/home/kali/Downloads/CIC_IDS_2018.csv"  # Change to your normal-only dataset
df = pd.read_csv(file_path)

# === Step 2: Keep Only Benign Traffic ===
df = df[df['Label'] == 'BENIGN']  # Adjust label if needed

# === Step 3: Preprocess ===
df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
df.fillna(0, inplace=True)

# === Step 4: Scale ===
scaler = StandardScaler()
X_train = scaler.fit_transform(df)

# === Step 5: Build Autoencoder ===
input_dim = X_train.shape[1]
encoding_dim = 14  # Bottleneck

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(64, activation='relu')(input_layer)
encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)

decoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(64, activation='relu')(decoded)
decoded = tf.keras.layers.Dense(input_dim, activation=None)(decoded)

autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# === Step 6: Train Model ===
history = autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=64,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# === Step 7: Save Model and Scaler ===
autoencoder.save("autoencoder_model.h5")
joblib.dump(scaler, "autoencoder_scaler.joblib")

print("✅ Autoencoder trained and saved.")
