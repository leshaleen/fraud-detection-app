import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go

# LOAD MODEL
model = joblib.load("real_project/model/fraud_model.pkl")
scaler = joblib.load("real_project/model/scaler.pkl")

st.set_page_config(layout="wide")

st.title("💳 Fraud Detection System")

# -------------------------------
# SESSION STATE FOR HISTORY
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# INPUT SECTION
# -------------------------------
st.header("🔍 Single Transaction Check")

step = st.number_input("Step", value=1)

type_map = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4
}

type_label = st.selectbox("Transaction Type", list(type_map.keys()))
type_val = type_map[type_label]

amount = st.number_input("Amount", value=1000.0)
oldbalanceOrg = st.number_input("Old Balance Sender", value=1000.0)
newbalanceOrig = st.number_input("New Balance Sender", value=500.0)
oldbalanceDest = st.number_input("Old Balance Receiver", value=0.0)
newbalanceDest = st.number_input("New Balance Receiver", value=500.0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Check Fraud"):

    data = np.array([[step, type_val, amount,
                      oldbalanceOrg, newbalanceOrig,
                      oldbalanceDest, newbalanceDest]])

    scaled = scaler.transform(data)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1] * 100

    # DISPLAY RESULT
    if pred == 1:
        st.error(f"🚨 Fraud Detected ({prob:.2f}%)")
    else:
        st.success(f"✅ Legit Transaction ({100 - prob:.2f}%)")

    # SAVE HISTORY
    st.session_state.history.append({
        "Type": type_label,
        "Amount": amount,
        "Result": "Fraud" if pred == 1 else "Legit",
        "Probability": round(prob, 2)
    })

    # GAUGE CHART
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Fraud Probability %"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig)

# -------------------------------
# HISTORY SECTION
# -------------------------------
st.header("📊 Prediction History")

if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist)
else:
    st.write("No predictions yet.")

# -------------------------------
# FILE UPLOAD SECTION
# -------------------------------
st.header("📂 Upload CSV for Batch Testing")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview:")
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):

        df["type"] = df["type"].astype("category").cat.codes

        X = df[[
            "step", "type", "amount",
            "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"
        ]]

        scaled = scaler.transform(X)
        preds = model.predict(scaled)

        df["Prediction"] = preds

        st.success("✅ Batch Prediction Completed")
        st.dataframe(df.head())
