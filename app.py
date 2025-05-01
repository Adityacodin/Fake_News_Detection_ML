import streamlit as st
import joblib
from scipy.sparse import hstack
import plotly.graph_objects as go

# Load saved model and transformers
model = joblib.load("Log_reg_model.pkl")
tfidf = joblib.load("tfidf.pkl")
subject_encoder = joblib.load("subject_encoder.pkl")

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# App title
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Use this tool to detect whether news articles are <b>fake</b> or <b>real</b>.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#90A4AE; font-size: 13px;'>Can detect any global, political, tech news from 2015 to 2017</p>", unsafe_allow_html=True)
st.markdown("---")

# Inputs
title = st.text_input("📝 Enter News Title")
text = st.text_area("📰 Enter News Text")
subject = st.selectbox("🗂️ Select Subject", subject_encoder.categories_[0].tolist())

# Predict
if st.button("🔍 Predict"):
    if not title.strip() and not text.strip():
        st.warning("Please enter at least Title or Text!")
    else:
        combined = title + " - " + text if title and text else title or text
        X_text = tfidf.transform([combined])
        X_subject = subject_encoder.transform([[subject]])
        X_final = hstack([X_text, X_subject])

        pred = model.predict(X_final)[0]
        proba = model.predict_proba(X_final)[0]
        confidence = max(proba) * 100

        # Show prediction
        st.markdown("---")
        if pred == 1:
            st.success("✅ The news is **REAL**.")
        else:
            st.error("❌ The news is **FAKE**.")

        # Speedometer-style confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Confidence Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "#ff4d4d"},
                    {'range': [40, 70], 'color': "#ffcc00"},
                    {'range': [70, 100], 'color': "#28a745"}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)
