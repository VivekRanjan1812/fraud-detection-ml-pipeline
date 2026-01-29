import streamlit as st
import pandas as pd
from src.model import predict

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Credit Card Fraud Detection")

st.markdown("""
Upload a CSV file containing transaction data.  
The app will predict potential fraudulent transactions.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Predict button
    if st.button("Predict Fraud"):
        try:
            predictions = predict(df)
            df['Prediction'] = predictions
            st.subheader("Predictions")
            st.dataframe(df)

            # Download predictions
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")
