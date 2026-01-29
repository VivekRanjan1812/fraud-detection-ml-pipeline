import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.preprocess import load_and_clean


def add_features(df):
    """
    Add new meaningful features
    """

    df = df.copy()

    # 1. Convert Time (seconds) to Hour of Day
    df["Hour"] = (df["Time"] / 3600) % 24

    # 2. Log transform Amount
    df["Log_Amount"] = np.log1p(df["Amount"])

    # 3. Z-score of Amount
    df["Amount_Z"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    # 4. Rolling Mean of Amount
    df["Amount_Rolling_Mean"] = (
        df["Amount"]
        .rolling(window=5, min_periods=1)
        .mean()
    )

    return df


def main():
    df = load_and_clean()
    df = add_features(df)

    print("✅ Feature Engineering Completed")
    print("Shape:", df.shape)
    print("New Columns:", df.columns[-4:])


if __name__ == "__main__":
    main()
