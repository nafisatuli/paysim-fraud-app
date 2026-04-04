import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# Load model + threshold
# ─────────────────────────────────────────────────────────────
model     = joblib.load("paysim_fraud_model.pkl")
threshold = joblib.load("threshold.pkl")

# DEBUG — remove after fixing
# st.write("Model features:", model.feature_names_in_.tolist())
# st.write("Threshold:", threshold)

# EXACT feature order from X.columns in notebook:
# ['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest',
#  'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER',
#  'amount_to_balance', 'is_zero_balance']
FEATURE_ORDER = [
    "step", "amount",
    "oldbalanceOrg", "oldbalanceDest",
    "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    "amount_to_balance", "is_zero_balance"
]

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Mobile Money Fraud Detection System")

tab1, tab2 = st.tabs(["🔍 Predict Transaction", "📊 Model Leaderboard"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ════════════════════════════════════════════════════════════
with tab1:
    st.write("Enter transaction details below:")

    step             = st.number_input("Transaction Step (Time Index)", min_value=0, value=1)
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_IN", "CASH_OUT", "DEBIT"]
    )
    amount           = st.number_input("Transaction Amount",   min_value=0.0, value=0.0)
    oldbalanceOrg    = st.number_input("Sender Old Balance",   min_value=0.0, value=0.0)
    oldbalanceDest   = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
    newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=0.0)

    st.caption("ℹ️ Post-transaction balances are excluded to prevent data leakage in the model.")

    if st.button("Predict Fraud Risk"):

        # ── Engineered features (same as training) ─────────────────────────
        amount_to_balance = min(amount / (oldbalanceOrg + 1), 1.0)
        is_zero_balance   = 1 if oldbalanceOrg == 0 else 0

        # ── One-hot encoding (CASH_IN dropped as reference) ────────────────
        type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
        type_DEBIT    = 1 if transaction_type == "DEBIT"    else 0
        type_PAYMENT  = 1 if transaction_type == "PAYMENT"  else 0
        type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0

        # ── Build DataFrame and reindex to exact column order ──────────────
        input_df = pd.DataFrame([{
            "step":              step,
            "amount":            amount,
            "oldbalanceOrg":     oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,   
            "oldbalanceDest":    oldbalanceDest,
            "newbalanceDest": newbalanceDest,  
            "type_CASH_OUT":     type_CASH_OUT,
            "type_DEBIT":        type_DEBIT,
            "type_PAYMENT":      type_PAYMENT,
            "type_TRANSFER":     type_TRANSFER,
            "amount_to_balance": amount_to_balance,
            "is_zero_balance":   is_zero_balance,
        }])

        input_df = input_df.reindex(columns=FEATURE_ORDER, fill_value=0)

        # ── Predict ────────────────────────────────────────────────────────
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        st.metric("Fraud Probability", f"{probability:.4f}")

        # ── Gauge chart ────────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 2),
            number={"suffix": "%"},
            title={"text": "Fraud Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "crimson" if probability > threshold else "green"},
                "steps": [
                    {"range": [0,  40], "color": "#d4edda"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100],"color": "#f8d7da"},
                ],
                "threshold": {
                    "line":      {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value":     threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        if probability > threshold:
            st.error("⚠️ High Fraud Risk Detected")
        else:
            st.success("✅ Low Fraud Risk")

# ════════════════════════════════════════════════════════════
# TAB 2 — Model Leaderboard
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Comparison")
    st.caption("Metrics evaluated on the 20% held-out test set (200,000 transactions).")

    results_df = pd.DataFrame({
        "Model":     ["Logistic Regression", "Random Forest", "XGBoost", "Isolation Forest"],
        "Accuracy":  [0.936855, 0.999975, 0.999295, 0.989425],
        "Precision": [0.019640, 0.980620, 0.645833, 0.021091],
        "Recall":    [1.000000, 1.000000, 0.980237, 0.162055],
        "F1 Score":  [0.038523, 0.990215, 0.778650, 0.037324],
    })

    # ── Styled table ──────────────────────────────────────────────────────
    st.markdown("#### 📋 Results Table")

    def highlight_best(s):
        is_best = s == s.max()
        return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_best]

    styled = (
        results_df.set_index("Model")
        .style
        .apply(highlight_best, axis=0)
        .format("{:.4f}")
    )
    st.dataframe(styled, use_container_width=True)

    # ── Grouped bar chart ─────────────────────────────────────────────────
    st.markdown("#### 📊 Visual Comparison")

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors  = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    fig_bar = go.Figure()
    for metric, color in zip(metrics, colors):
        fig_bar.add_trace(go.Bar(
            name=metric,
            x=results_df["Model"],
            y=results_df[metric],
            marker_color=color,
            text=results_df[metric].round(3),
            textposition="outside",
        ))

    fig_bar.update_layout(
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.12]),
        xaxis_title="Model",
        legend_title="Metric",
        height=480,
        template="plotly_white",
        title="Accuracy / Precision / Recall / F1 — All Models",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Key takeaway ──────────────────────────────────────────────────────
    st.info(
        "**Key Takeaway:** Random Forest achieves the highest overall performance "
        "(F1 = 0.990, Recall = 1.000), making it the most reliable model for this task. "
        "Logistic Regression achieves perfect recall but very low precision, flagging too many "
        "legitimate transactions as fraud. "
        "Isolation Forest performs poorly in the supervised setting due to its unsupervised nature."
    )
