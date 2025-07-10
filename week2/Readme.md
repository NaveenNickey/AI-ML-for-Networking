# Week 2: Threat Detection in Network Traffic

This module focuses on detecting specific types of cyber threats using behavioral features and multi-class classification models.

---

## Contents

- `dataset/` - Contains labeled threat datasets (e.g., CIC-IDS-2018)
- `threat_detection.py` - Script for training a threat detection model
- `model.pkl` - Trained model file
- `scaler.pkl` - Feature scaler

---

## How to Run

1. Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn lightgbm matplotlib joblib
    ```

2. Execute the script:
    ```bash
    python3 CIC_IDS_2018.py
    ```

3. The script will:
    - Preprocess and normalize the data
    - Train a LightGBM classifier on labeled threats
    - Save model and scaler for use in stacked models
    - Output confusion matrix and classification report

---

## Notes

- Works with multi-class labels (e.g., DoS, Botnet, BruteForce).
- Update file paths if using a custom dataset.
