import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import RAW_DATA_PATH


def clean_data(df):
    """
    Remove duplicates and missing values
    """

    df = df.drop_duplicates()
    df = df.dropna()

    return df


def load_and_clean():
    """
    Load raw data and clean it
    """

    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_data(df)

    return df


def main():
    df = load_and_clean()
    print("✅ Cleaned Data Shape:", df.shape)


if __name__ == "__main__":
    main()
