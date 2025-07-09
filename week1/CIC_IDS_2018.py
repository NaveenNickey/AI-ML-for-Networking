import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# === Step 1: Load Dataset ===
file_path = "/home/kali/Downloads/CIC-IDS_2018.csv"  # <-- Update as needed
df = pd.read_csv(file_path)

# === Step 2: Automatically detect label column (fallback to user input) ===
possible_label_cols = [col for col in df.columns if 'label' in col.lower() or 'attack' in col.lower()]
if possible_label_cols:
    label_col = possible_label_cols[0]
else:
    raise ValueError("❌ Couldn't find label column automatically. Please specify manually.")

print(f"✅ Using label column: {label_col}")

# === Step 3: Encode labels ===
df = df.dropna(subset=[label_col])  # Drop rows where label is missing
df[label_col] = df[label_col].astype(str)  # Ensure string type for mapping
label_map = {label: idx for idx, label in enumerate(df[label_col].unique())}
df[label_col] = df[label_col].map(label_map)
print(f"✅ Label mapping: {label_map}")

# === Step 4: Drop non-numeric columns ===
non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
df = df.drop(columns=non_numeric_cols)  # Drop all object columns

# === Step 5: Replace Infinite & NaNs ===
X = df.drop(label_col, axis=1)
y = df[label_col]
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# === Step 6: Feature Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 7: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 8: LightGBM Dataset ===
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# === Step 9: LightGBM Params ===
params = {
    'objective': 'multiclass',
    'num_class': len(label_map),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbosity': -1
}

# === Step 10: Train Model ===
print("\n🚀 Training LightGBM model...\n")
model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# === Step 11: Predict & Evaluate ===
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred_classes))

# === Step 12: Save Model ===
model_filename = "lightgbm_traffic_model.pkl"
joblib.dump(model, model_filename)
print(f"\n✅ Model saved as '{model_filename}'")

# === Step 13: Feature Importance Plot ===
lgb.plot_importance(model, max_num_features=15)
plt.title("Top 15 Important Features")
plt.tight_layout()
plt.show()
