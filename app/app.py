from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
import shap
import plotly.graph_objects as go
import time

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Fraud Detection", page_icon="💳", layout="wide")

# ================== CSS ==================
st.markdown("""
<style>
.block-container {
    max-width: 900px;
    margin: auto;
}
.stApp {
    background: linear-gradient(135deg, #020617, #0F172A);
}
.title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #00FFA3;
}
div.stButton {
    display: flex;
    justify-content: center;
}
div.stButton > button {
    background: linear-gradient(135deg, #2563EB, #06B6D4);
    color: white;
    font-size: 18px;
    border-radius: 14px;
    padding: 14px 32px;
    border: none;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    margin-top: 30px;
    text-align: center;
}
section[data-testid="stSidebar"] {
    background: #020617 !important;
}
[data-baseweb="input"] input {
    background-color: #111827 !important;
    color: white !important;
}
.js-plotly-plot {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, "fraud_model.h5"))

# ================== TITLE ==================
st.markdown("<div class='title'>AI Fraud Detection System</div>", unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.header("⚙️ Raw Transaction Input")

amount = st.sidebar.number_input("Amount", value=100.0)
time_val = st.sidebar.number_input("Time", value=10000.0)

raw_features = []
for i in range(28):
    val = st.sidebar.number_input(f"Raw Feature {i+1}", value=0.0)
    raw_features.append(val)

# ================== PCA PIPELINE ==================
raw_array = np.array(raw_features).reshape(1, -1)

scaler = StandardScaler()
scaled = scaler.fit_transform(raw_array)

pca = PCA(n_components=28)
dummy_data = np.random.rand(50, len(raw_features))
pca.fit(dummy_data)

pca_transformed = pca.transform(scaled)

# ================== BUTTON ==================
detect = st.button("🚀 Detect Fraud")

# ================== MAIN ==================
if detect:

    # ✅ FINAL INPUT (STEP 4 FIX)
    final_input = []

    final_input.append(time_val)

    for val in pca_transformed[0]:
        final_input.append(val)

    final_input.append(amount)

    final_input = np.array(final_input).reshape(1, -1)

    with st.spinner("Analyzing..."):
        time.sleep(1)
        prediction = model.predict(final_input)
        prob = float(prediction[0][0])

    # RESULT
    if prob > 0.5:
        st.markdown(f"<div class='card'>🚨 Fraud Detected<br>Confidence: {prob:.2f}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='card'>✅ Legitimate<br>Confidence: {1-prob:.2f}</div>", unsafe_allow_html=True)

    # ================== GAUGE ==================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Fraud Probability"},
        gauge={'axis': {'range': [0,1]}}
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================== SHAP ==================
    st.subheader("📊 Feature Importance")

    try:
        background = np.random.normal(0,1,(5,30))
        explainer = shap.Explainer(model, background)
        shap_values = explainer(final_input)

        shap_vals = shap_values.values[0]

        fig_bar = go.Figure([go.Bar(
            x=shap_vals,
            y=[f"F{i}" for i in range(len(shap_vals))],
            orientation='h'
        )])

        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    except:
        st.warning("Feature explanation not available")

# ================== CSV ==================
st.sidebar.subheader("📁 Upload CSV")

file = st.sidebar.file_uploader("Upload file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.write("Preview")
    st.dataframe(df.head())

    if st.button("Run Batch Detection"):
        preds = model.predict(df.values)
        df['Fraud'] = preds > 0.5
        st.dataframe(df.head())

        









