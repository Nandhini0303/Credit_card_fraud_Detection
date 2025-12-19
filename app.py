
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load Model and Scaler
# -------------------------------
st.title("💳 Credit Card Fraud Detection Dashboard")

try:
    model = joblib.load("xgboost_fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("Model and Scaler Loaded Successfully!")
except:
    st.error("Model or Scaler not found! Please place them in the same folder as app.py")

# -------------------------------
# Input Form for all features
# -------------------------------
st.header("Enter Transaction Details")

feature_names = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

inputs = {}

with st.form("fraud_form"):
    for f in feature_names:
        inputs[f] = st.number_input(f"Enter value for {f}", value=0.0)
    
    submit = st.form_submit_button("Predict")

# -------------------------------
# Prediction
# -------------------------------
if submit:
    input_df = pd.DataFrame([inputs])
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔍 Prediction Result")
    
    if pred == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Valid")
    
    st.write(f"**Fraud Probability:** {proba:.4f}")


