# 🔐 Intel Internship Report – AI/ML For Networking

This folder contains the summary and documentation of the project completed during the internship at **Intel**. The aim of the project was to build a scalable and intelligent AI-powered system to classify, detect, and analyze network traffic in real-time with a focus on cybersecurity threat prevention and anomaly identification.

---

## 🎯 Project Objective

The project was designed to solve three key cybersecurity tasks: classify types of traffic (such as DNS, HTTP, VPN), detect known cyber threats (like DDoS and malware), and identify unknown or abnormal behavior using anomaly detection. All components were later unified using a stacked ensemble model to build a robust AI security system.

---

## 📚 Dataset Formation

Multiple benchmark datasets were utilized across various components. The CIC-IDS 2018 and UNSW-NB15 datasets were used for supervised traffic classification and threat detection tasks. The TON-IoT dataset was used to train an unsupervised anomaly detection model. Additional datasets such as ISCX VPN-nonVPN and AppTraffic were referenced for privacy-preserving and extended classification capabilities.

---

## 🛠️ Feature Engineering

Feature extraction was performed using tools like CICFlowMeter and Wireshark to extract flow-level features. Additional deep-packet inspection was used to analyze TLS handshakes for encrypted traffic analysis. These extracted features were essential to building accurate and lightweight machine learning models.

---

## ⚙️ Data Preprocessing

All datasets underwent preprocessing which included cleaning missing values, encoding categorical variables, and normalizing numerical data using `pandas` and `scikit-learn`. Balanced datasets were critical to reduce bias and ensure model fairness, especially in the case of imbalanced attack class distributions.

---

## 🧠 Model Building

Three main types of models were developed during the internship. For traffic classification, models like ID-INN, LSTM, and LightGBM were tested, with LightGBM selected for its high performance and speed. For threat detection, models including CNN, Decision Trees, and Isolation Forest were explored to detect DDoS, brute force, and malware attacks. The anomaly detection module was built using an Autoencoder in TensorFlow trained only on normal traffic, with high reconstruction error used to flag anomalies.

---

## 🧩 Meta-Model Integration

The predictions from the traffic classification, threat detection, and anomaly detection models were combined and used as inputs to a stacked ensemble meta-model. The meta-model was trained using logistic regression, allowing the system to benefit from the strengths of each individual model. The integration pipeline included saving outputs in `x_test.csv` and corresponding true labels in `y_meta.csv`.

---

## 🚀 Real-Time Inference Pipeline

A real-time inference system was simulated and built using Streamlit to provide an interactive dashboard. This dashboard allowed users to upload data, view predictions from all models, and observe how the system handled real-time traffic classification and threat detection tasks.

---

## 🔧 Optimization

Throughout the project, performance optimization was a priority. Each model was tuned for both accuracy and latency. LightGBM was selected for its fast training and execution. TensorFlow models were designed with efficiency in mind, ensuring the system could be extended for use in near real-time deployment environments.

---

## 🛡️ Privacy-Preserving Design

Special attention was given to preserving user privacy, particularly in handling encrypted traffic such as VPN flows. Models were designed to ensure no sensitive data was exposed, and encrypted flows were handled using non-invasive techniques like TLS handshake analysis.

---

## 📅 Weekly Progress Snapshot

### 📆 Week 1: Traffic Classification
- Downloaded and explored the CIC-IDS 2018 dataset.
- Performed feature extraction using CICFlowMeter.
- Preprocessed and cleaned the dataset using Pandas.
- Trained baseline models like Random Forest and XGBoost.
- Evaluated performance with a goal of ≥90% accuracy.
- Started development of a Streamlit dashboard for model visualization.

### 📆 Week 2: Threat Detection
- Integrated the UNSW-NB15 dataset for detecting cyber threats.
- Preprocessed and normalized the dataset.
- Trained CNN, Decision Trees, and Isolation Forest models.
- Evaluated precision, recall, and F1-score for multiple attack classes.
- Exported `threat_model.pkl` and scaler files.
- Updated the dashboard to include threat detection visualization.

### 📆 Week 3: Anomaly Detection
- Switched to the TON-IoT dataset focusing on unsupervised learning.
- Trained an autoencoder using TensorFlow only on normal traffic.
- Created scripts for saving `autoencoder_model.h5` and `scaler.joblib`.
- Tested anomaly predictions on malicious traffic to evaluate detection accuracy.
- Fine-tuned reconstruction error threshold to reduce false positives.

### 📆 Week 4: Model Integration & Finalization
- Combined predictions from all three base models into `x_test.csv`.
- Generated `y_meta.csv` with corresponding ground truth.
- Trained the final stacked meta-model using logistic regression.
- Evaluated system-wide performance and exported `meta_model.pkl`.
- Finalized Streamlit dashboard with all model outputs integrated.
- Recorded demo video and completed documentation and report.

---

## 📝 Notes & Final Thoughts

All model files (`.pkl`, `.h5`, `.joblib`) are essential for the complete working pipeline. The meta-model requires base model predictions as inputs and corresponding labels. TensorFlow models were tested with Python ≤3.12 and are compatible with Google Colab if required. The modular structure of this project ensures that improvements in one component do not break the rest of the pipeline, making it scalable and extensible.

---

## ✅ Deliverables

- Trained model files: `.pkl`, `.h5`, `.joblib`  
- Meta-model input and labels: `x_test.csv`, `y_meta.csv`  
- Real-time inference dashboard (Streamlit)  
- Complete source code with documentation  
- Final PDF report  
- Architecture Diagram  
- Video Demonstration

---

> 📌 For visual reference, check out the [Architecture Diagram](../Architecture%20Diagram).  
> 🎥 For a working walkthrough, see the [Video Demonstration](../Video%20Demonstration).
