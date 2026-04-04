import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# Load model + threshold + feature order
# ─────────────────────────────────────────────────────────────
model = joblib.load("paysim_fraud_model.pkl")
threshold = joblib.load("threshold.pkl")
try:
    feature_order = joblib.load("features.pkl")
except:
    feature_order = [
        "step", "amount",
        "oldbalanceOrg", "oldbalanceDest",
        "amount_to_balance", "is_zero_balance",
        "type_CASH_OUT", "type_DEBIT",
        "type_PAYMENT", "type_TRANSFER"
    ]

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Mobile Money Fraud Detection System")

# Tabs
tab1, tab2 = st.tabs(["🔍 Predict Transaction", "📊 Model Leaderboard"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ════════════════════════════════════════════════════════════
with tab1:
    st.write("Enter transaction details below:")

    step = st.number_input("Transaction Step (Time Index)", min_value=0, value=1)

    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_IN", "CASH_OUT", "DEBIT"]
    )

    amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0)
    oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=0.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)

    if st.button("Predict Fraud Risk"):

        # ── Feature Engineering (same as training) ──
        amount_to_balance = amount / (oldbalanceOrg + 1)
        is_zero_balance = 1 if oldbalanceOrg == 0 else 0

        # ── One-hot encoding (drop_first=True → CASH_IN dropped) ──
        type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
        type_DEBIT = 1 if transaction_type == "DEBIT" else 0
        type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
        type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0

        # ── Create DataFrame (IMPORTANT FIX) ──
        input_df = pd.DataFrame([{
            "step": step,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "oldbalanceDest": oldbalanceDest,
            "amount_to_balance": amount_to_balance,
            "is_zero_balance": is_zero_balance,
            "type_CASH_OUT": type_CASH_OUT,
            "type_DEBIT": type_DEBIT,
            "type_PAYMENT": type_PAYMENT,
            "type_TRANSFER": type_TRANSFER
        }])

        # ── Ensure correct feature order ──
        input_df = input_df.reindex(columns=feature_order, fill_value=0)
        # ── Predict ──
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        st.metric("Fraud Probability", f"{probability:.4f}")

        # ── Gauge Chart ──
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 2),
            number={"suffix": "%"},
            title={"text": "Fraud Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "crimson" if probability > threshold else "green"},
                "steps": [
                    {"range": [0, 40], "color": "#d4edda"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100
                }
            }
        ))

        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Decision ──
        if probability > threshold:
            st.error("⚠️ High Fraud Risk Detected")
        else:
            st.success("✅ Low Fraud Risk")


# ════════════════════════════════════════════════════════════
# TAB 2 — Model Leaderboard
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Comparison")

    results_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.9368, 0.9999, 0.9992],
        "Precision": [0.0196, 0.9806, 0.6458],
        "Recall": [1.0000, 1.0000, 0.9802],
        "F1 Score": [0.0385, 0.9902, 0.7786],
    })

    st.dataframe(results_df, use_container_width=True)

    fig_bar = go.Figure()
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for metric in metrics:
        fig_bar.add_trace(go.Bar(
            name=metric,
            x=results_df["Model"],
            y=results_df[metric]
        ))

    fig_bar.update_layout(
        barmode="group",
        title="Model Comparison",
        yaxis=dict(range=[0, 1.1])
    )

    st.plotly_chart(fig_bar, use_container_width=True)
