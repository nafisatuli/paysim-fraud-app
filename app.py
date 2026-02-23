import streamlit as st
import joblib
import numpy as np

model = joblib.load("paysim_fraud_model.pkl")
threshold = joblib.load("threshold_paysim.pkl")

st.title("💳 Mobile Money Fraud Detection System")

st.write("Enter transaction details below:")

# Basic inputs
step = st.number_input("Transaction Step (Time Index)", min_value=0, value=1)

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_IN", "CASH_OUT", "DEBIT"]
)

amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0)
newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0)
oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0)
newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0)

if st.button("Predict Fraud Risk"):

    # Engineered features
    balanceDiffOrig = oldbalanceOrg - newbalanceOrig
    balanceDiffDest = newbalanceDest - oldbalanceDest
    errorBalanceOrig = amount - balanceDiffOrig
    errorBalanceDest = amount - balanceDiffDest

    # One-hot encoding (must match training)
    type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
    type_DEBIT = 1 if transaction_type == "DEBIT" else 0
    type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
    type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0
    # CASH_IN was dropped due to drop_first=True

    input_data = np.array([[
        step,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        balanceDiffOrig,
        balanceDiffDest,
        errorBalanceOrig,
        errorBalanceDest,
        type_CASH_OUT,
        type_DEBIT,
        type_PAYMENT,
        type_TRANSFER
    ]])

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Fraud Probability: {probability:.4f}")

    if probability > threshold:
        st.error("⚠ High Fraud Risk Detected")
    else:
        st.success("✅ Low Fraud Risk")