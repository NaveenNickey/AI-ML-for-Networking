# Week 1: Network Traffic Classification

This module focuses on building a supervised machine learning model to classify types of network traffic using labeled datasets.

---

## Contents

- `dataset/` - Contains labeled traffic datasets
- `traffic_classifier.py` - Python script to train and test the classifier
- `model.pkl` - Saved trained model
- `scaler.pkl` - Saved scaler for normalization

---

## How to Run

1. Make sure required libraries are installed:
    ```bash
    pip install pandas numpy scikit-learn lightgbm matplotlib joblib
    ```

2. Run the classification script:
    ```bash
    python3 CIC_IDS_2018.py
    ```

3. This will:
    - Load and preprocess the dataset
    - Train a LightGBM classifier
    - Save the model and scaler for reuse
    - Print performance metrics (accuracy, classification report)

---

## Notes

- Modify the dataset path inside `CIC_IDS_2018.py` to use your own data.
- The model supports multi-class classification for different traffic types.
