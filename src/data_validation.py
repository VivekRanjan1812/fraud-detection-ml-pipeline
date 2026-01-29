import pandas as pd
from config import RAW_DATA_PATH


def load_data():
    """Load raw dataset"""
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def validate_data(df):
    """Check data quality"""

    print("🔍 Data Validation Report")

    # Shape
    print(f"Shape: {df.shape}")

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Target distribution
    print("\nTarget Distribution:")
    print(df["Class"].value_counts())


def main():
    df = load_data()
    validate_data(df)


if __name__ == "__main__":
    main()
