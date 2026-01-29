import pandas as pd
import joblib
from src.preprocess import preprocess   # use your existing preprocessing

# Load trained model
model = joblib.load("models/fraud_model.pkl")

def predict(df: pd.DataFrame):
    """
    Takes a dataframe, applies preprocessing, and returns predictions
    """
    df_processed = preprocess(df)  # your existing preprocessing function
    predictions = model.predict(df_processed)
    return predictions
