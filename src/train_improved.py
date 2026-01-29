import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.features import add_features
from src.preprocess import load_and_clean
from src.config import RANDOM_STATE, TEST_SIZE


def prepare_data():

    df = load_and_clean()
    df = add_features(df)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def train_with_smote(X, y):

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ))
    ])

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    roc_scores = []
    f1_scores = []

    print("🚀 Training with SMOTE...")

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

    print("\n📊 Improved Average:")
    print("ROC-AUC:", np.mean(roc_scores))
    print("F1:", np.mean(f1_scores))

    return pipeline


def main():

    X, y = prepare_data()

    model = train_with_smote(X, y)

    print("\n🎉 Improved Training Done!")


if __name__ == "__main__":
    main()
