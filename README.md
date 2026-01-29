# 💳 Credit Card Fraud Detection — ML Engineer Assignment

**Author:** Vivek Ranjan (MCA, UPES)

---

#  1. Project Overview

This project implements a **production-grade machine learning pipeline** for detecting fraudulent credit card transactions.

It covers the complete ML lifecycle, including:

* Data validation
* Feature engineering
* Model training
* Debugging
* Optimization
* Deployment readiness

---

#  2. Dataset

**Source:** Credit Card Fraud Dataset (Kaggle)

* **Total Samples:** 284,807
* **Fraud Cases:** 492 (Highly Imbalanced)
* **Target Column:** `Class`

| Value | Meaning                |
| ----- | ---------------------- |
| 0     | Normal Transaction     |
| 1     | Fraudulent Transaction |

---

#  3. Folder Structure

```
fraud-detection-ml/
│
├── data/
├── models/
├── src/
├── README.md
└── requirements.txt
```

---

#  4. Task 1 — ML Pipeline & Data Validation

##  Data Validation

* Checked missing values
* Removed duplicate entries
* Verified data types

##  Feature Engineering

Four additional features were created:

* `Hour`
* `Log_Amount`
* `Amount_Z`
* `Amount_Rolling_Mean`

##  Model Selection

**Model Used:** Logistic Regression

### Reason for Selection:

* Easy to interpret
* Stable performance
* Suitable for imbalanced datasets

##  Cross Validation

* Stratified K-Fold (5 folds)

##  Evaluation Metrics

* ROC-AUC Score
* F1 Score
* Precision
* Recall

##  Model Saving

* Model stored using `joblib`

##  Reproducibility

* Random seed fixed for consistent results

---

#  5. Task 2 — Debugging

##  Problems Identified

* Low F1 score
* Severe class imbalance
* Unstable predictions

## 🔍 Root Causes

* Skewed data distribution
* Default probability threshold
* Random sampling effects

##  Fixes Applied

* SMOTE (Synthetic Minority Oversampling)
* Threshold tuning

##  Results

**Before Optimization:**

* F1 Score ≈ 0.07

**After Optimization:**

* F1 Score ≈ 0.10

---

#  6. Task 3 — Performance Improvement

##  Techniques Used

* SMOTE
* Advanced feature engineering
* Threshold optimization

##  Improvement Achieved

* 40%+ increase in F1 Score

##  Reason for Improvement

* Better learning of minority class patterns
* Reduced bias toward majority class

---

#  7. Task 4 — System Design

##  Architecture

```
Transaction Data
      ↓
Data Ingestion
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Storage
      ↓
Inference API
      ↓
Fraud Alerts
      ↓
Monitoring
      ↓
Retraining
```

##  Monitoring

* Performance tracking
* Data drift detection

##  Retraining Strategy

* Monthly retraining
* Automatic retraining on drift detection

---

#  8. Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Joblib

---

# ▶ 9. How to Run the Project

##  Installation

```bash
pip install -r requirements.txt
```

## ▶ Execution

```bash
python src/preprocess.py
python src/features.py
python src/train.py
python src/evaluate.py
python src/threshold_tuning.py
```

---

#  10. Conclusion

This project demonstrates a **complete end-to-end machine learning lifecycle**, from data preprocessing to deployment readiness.

It highlights practical handling of imbalanced datasets and production-level ML design principles.

---

#  Fraud Detection ML App

This project is deployed as a web application for real-time fraud detection.

**Live App:**
👉 [https://fraud-detection-ml-pipeline-my6gqpagn4lbg3iardrkcy.streamlit.app/](https://fraud-detection-ml-pipeline-my6gqpagn4lbg3iardrkcy.streamlit.app/)

---

✨ *Developed by Vivek Ranjan (MCA, UPES)*


