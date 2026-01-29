import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")

# Random seed (for reproducibility)
RANDOM_STATE = 42

# Test size
TEST_SIZE = 0.2
