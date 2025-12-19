
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load model
# -------------------------------
model = joblib.load("model_xgb.pkl")

st.title("💳 Credit Card Fraud Detection App")
st.write("Use this app to detect fraud transactions using XGBoost model.")

# ================================================================
#                1.   Single Transaction Prediction
# ================================================================
st.header("🔹 Single Transaction Prediction")

# Read input feature names from model training
feature_names = [
    "Time",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28",
    "Amount"
]

user_values = []

for feature in feature_names:
    val = st.number_input(f"Enter {feature}", value=0.0)
    user_values.append(val)

if st.button("Predict Single Transaction"):
    input_array = np.array(user_values).reshape(1, -1)
    pred = model.predict(input_array)[0]

    st.subheader("🔍 Prediction Result:")
    if pred == 1:
        st.error("🚨 Fraud Detected!")
    else:
        st.success("✅ Normal Transaction")


# ================================================================
#                2.   BULK CSV Prediction
# ================================================================
st.header("📁 Upload CSV for Bulk Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Check if all required columns exist
    missing = [col for col in feature_names if col not in df.columns]

    if missing:
        st.error(f"❌ Missing columns in your CSV: {missing}")
    else:
        # Predict
        predictions = model.predict(df[feature_names])

        df["Prediction"] = predictions
        df["Result"] = df["Prediction"].apply(lambda x: "Fraud" if x == 1 else "Normal")

        st.success("✅ Prediction Completed!")

        st.write("### 🔍 Prediction Output")
        st.dataframe(df)

        # Download predictions
        csv_output = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Prediction File",
            data=csv_output,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

