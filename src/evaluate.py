import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, roc_auc_score

from src.features import add_features
from src.preprocess import load_and_clean
from src.config import MODEL_PATH


def evaluate_model():

    print("📥 Loading model...")
    model = joblib.load(MODEL_PATH)

    print("📊 Preparing test data...")

    df = load_and_clean()
    df = add_features(df)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Take last 20% as test (simulate unseen data)
    split = int(0.8 * len(df))

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    print("🧪 Evaluating model...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)

    print("\nROC-AUC:", roc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()
