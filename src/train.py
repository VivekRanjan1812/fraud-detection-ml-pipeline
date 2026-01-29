import sys
import os

# Fix path issue
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

from src.features import add_features
from src.preprocess import load_and_clean
from src.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE


def prepare_data():
    """
    Load, clean, and create features
    """

    df = load_and_clean()
    df = add_features(df)

    # Separate input and output
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def train_with_cv(X, y):
    """
    Train model using Cross Validation
    """

    # ML Pipeline (Scaler + Model)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    roc_scores = []
    f1_scores = []

    print("🚀 Starting Cross Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        roc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)

        roc_scores.append(roc)
        f1_scores.append(f1)

        print(f"Fold {fold} → ROC-AUC: {roc:.4f}, F1: {f1:.4f}")

    print("\n📊 Average Performance:")
    print("ROC-AUC:", np.mean(roc_scores))
    print("F1:", np.mean(f1_scores))

    return pipeline


def train_final_model(X, y, model):
    """
    Train final model and save it
    """

    print("\n💾 Training final model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print(f"✅ Model saved at: {MODEL_PATH}")

    return X_test, y_test


def main():

    print("📥 Preparing data...")
    X, y = prepare_data()

    print("📐 Training with Cross Validation...")
    model = train_with_cv(X, y)

    print("🏁 Training final model...")
    X_test, y_test = train_final_model(X, y, model)

    print("🎉 Training Completed!")


if __name__ == "__main__":
    main()
