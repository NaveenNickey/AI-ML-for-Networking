import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

def load_and_prepare_data(path, label_col='Label', drop_cols=None):
    """
    Load and preprocess any intrusion detection dataset.
    - path: CSV file path
    - label_col: Column containing the class labels
    - drop_cols: List of columns to drop (e.g., IP addresses, timestamps)
    """
    df = pd.read_csv(path)

    # Drop specified non-feature columns if they exist
    if drop_cols:
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    # Drop rows with missing or infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Separate features and labels
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' not found in dataset.So use the To_Find_The_Label.py program and then enter the label manually")
    y = df[label_col]
    X = df.drop(columns=[label_col])

    return X, y

def preprocess_features(X):
    """Standardize feature values."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def encode_labels(y):
    """Encode string labels to integers."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

def train_lgbm_model(X_train, y_train, num_classes):
    """Train LightGBM multiclass classifier."""
    model = LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        class_weight='balanced',
        metric='multi_logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate and display results."""
    y_pred = model.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== F1 Scores ===")
    print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
    print("Weighted F1:", f1_score(y_test, y_pred, average='weighted'))

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot top N feature importances."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx][:top_n], importances[sorted_idx][:top_n])
    plt.title(f"Top {top_n} Important Features")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.show()

# === MAIN EXECUTION ===
if __name__ == "__main__":
    dataset_path = "/home/kali/Downloads/AppTraffic.csv"  # <- Replace with your dataset
    drop_columns = ["Time","Source","No.","Destination","Protocol","Length","Info"]  # Customize per dataset
    label_column = 'Label'  # Make sure this matches the label column in your dataset

    try:
        # Load & preprocess
        X, y = load_and_prepare_data(dataset_path, label_column, drop_columns)
        X_scaled = preprocess_features(X)
        y_encoded, le = encode_labels(y)

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, stratify=y_encoded, test_size=0.3, random_state=42)
        model = train_lgbm_model(X_train, y_train, num_classes=len(le.classes_))

        # Evaluate
        evaluate_model(model, X_test, y_test, le)

        # Feature Importance
        plot_feature_importance(model, X.columns)

    except Exception as e:
        print("❌ ERROR:", e)
