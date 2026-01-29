import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib

from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.features import add_features
from src.preprocess import load_and_clean
from src.config import RANDOM_STATE, TEST_SIZE


def main():

    print("📥 Loading Data...")

    df = load_and_clean()
    df = add_features(df)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ))
    ])

    print("🚀 Training model...")

    pipeline.fit(X_train, y_train)

    y_probs = pipeline.predict_proba(X_test)[:, 1]

    print("\n🎯 Testing Thresholds...\n")

    best_f1 = 0
    best_t = 0

    for t in np.arange(0.01, 0.51, 0.02):

        y_pred = (y_probs >= t).astype(int)

        f1 = f1_score(y_test, y_pred)

        print(f"Threshold {t:.2f} → F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print("\n🏆 Best Threshold:", best_t)
    print("Best F1:", best_f1)

    final_pred = (y_probs >= best_t).astype(int)

    print("\n📊 Final Report:\n")

    print(classification_report(y_test, final_pred))


if __name__ == "__main__":
    main()
