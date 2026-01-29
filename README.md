 Credit Card Fraud Detection — ML Engineer Assignment

Author
Vivek Ranjan (MCA, UPES)

 1. Project Overview

This project implements a production-grade machine learning pipeline for detecting fraudulent credit card transactions.

It includes:
- Data validation
- Feature engineering
- Model training
- Debugging
- Optimization
- Deployment readiness


 2. Dataset

Source: Credit Card Fraud Dataset (Kaggle)

Samples: 284,807  
Fraud: 492 (Highly Imbalanced)

Target Column: Class  
0 = Normal  
1 = Fraud

 3. Folder Structure

fraud-detection-ml/
|
|-- data/
|-- models/
|-- src/
|-- README.md
|-- requirements.txt

 4. Task 1 — ML Pipeline Data Validation
- Checked missing values
- Removed duplicates
- Verified data types

Feature Engineering
Added 4 features:
- Hour
- Log_Amount
- Amount_Z
- Amount_Rolling_Mean

 Model
Logistic Regression

Reason:
- Interpretable
- Stable
- Good for imbalanced data

 Cross Validation
Stratified K-Fold (5 folds)

 Evaluation
- ROC-AUC
- F1 Score
- Precision
- Recall

Model Saving
Model saved using joblib

Reproducibility
Random seed fixed


5. Task 2 — Debugging

Problems
- Low F1 score
- Class imbalance
- Unstable predictions

Root Causes
- Data imbalance
- Default threshold
- Random sampling

### Fixes
- SMOTE
- Threshold tuning

### Results

Before:
F1 ≈ 0.07

After:
F1 ≈ 0.10

6. Task 3 — Performance Improvement

Techniques:
- SMOTE
- Feature engineering
- Threshold tuning

Improvement:
40%+ F1 increase

Reason:
Better minority learning


7. Task 4 — System Design

Architecture

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


Monitoring
- Performance tracking
- Drift detection

Retraining
- Monthly retraining
- Triggered on drift


8. Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Joblib

9. How to Run

Install:

pip install -r requirements.txt

Run:

python src/preprocess.py  
python src/features.py  
python src/train.py  
python src/evaluate.py  
python src/threshold_tuning.py  


 10. Conclusion

This project demonstrates a complete ML lifecycle from data processing to deployment readiness.
