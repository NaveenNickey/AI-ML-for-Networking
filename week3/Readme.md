# Week 3: Anomaly Detection with Autoencoder

This week introduces unsupervised anomaly detection using an autoencoder neural network trained on normal traffic to detect statistical outliers.

---

## Contents

- `dataset/` - Contains raw or preprocessed network traffic datasets
- `autoencoder_train.py` - Script to train the autoencoder
- `autoencoder_test.py` - Script to test and evaluate anomalies
- `autoencoder_model.h5` - Saved trained model
- `scaler.pkl` - Standard scaler used during training

---

## How to Run

1. Install required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib tensorflow joblib
    ```

2. Train the model:
    ```bash
    python3 autoencoder_train.py
    ```

3. Test the model:
    ```bash
    python3 autoencoder_test.py
    ```

---

## Notes

- Trained only on normal traffic to learn patterns.
- Anomalies are detected based on reconstruction error.
- Adjust threshold value in `autoencoder_test.py` for sensitivity control.
