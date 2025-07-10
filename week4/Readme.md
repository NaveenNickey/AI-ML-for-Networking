# Week 4: Final Meta Model (Stacking)

This module combines the three models (traffic classification, threat detection, anomaly detection) using stacking ensemble technique to improve overall performance.

---

## Contents

- `meta_model.pkl` - Final stacked model
- `scaler_*.pkl` - Scalers used by base models
- `model_*.pkl` - Base model files for classification, threat detection, anomaly flags
- `Stacked_meta_model.py` - Script to run the final meta model on input features

---

## How to Run

1. Install required packages:
    ```bash
    pip install pandas numpy scikit-learn lightgbm joblib
    ```

2. Run the stacked model:
    ```bash
    python3 Stacked_meta_model.py
    ```

3. The script will:
    - Load base model predictions as meta-features
    - Use the meta-model to predict final classification
    - Output results and accuracy metrics

---

## Notes

- All `.pkl` files must be in the same directory or update paths in the script.
- Input features are predictions from Week 1–3 models.
- Stacked model enhances overall classification accuracy and robustness.
