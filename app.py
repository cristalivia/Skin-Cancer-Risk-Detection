import domain_cleaner
from domain_cleaner import DomainCleaner

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Skin Cancer Risk Prediction",
    layout="centered"
)

model = load("skin_cancer_model.joblib")

def probability_to_risk(prob):
    score = int(round(prob * 10))
    return max(1, min(score, 10))

# =========================
# HEADER
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    background-color: #F5F5F5;
}

h1, h2, h3 {
    color: #2F4157;
}

p, li {
    color: #577C8E;
    font-size: 15px;
}

.section-card {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 14px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    margin-bottom: 24px;
}

.guidelines li {
    margin-bottom: 8px;
}

.result-box {
    background-color: #E0E5E8;
    padding: 20px;
    border-radius: 12px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<div class="section-card">
    <h1>🚨 Skin Cancer Risk Detection</h1>
    <p>
        This application uses a machine learning–based risk assessment model to estimate 
        an individual's potential risk of skin cancer based on demographic factors, 
        lifestyle habits, general health conditions, and medical history.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# INPUT FORM
# =========================
with st.form("risk_form"):

    st.subheader("👤 Demographics")
    sex = st.selectbox(
        "Sex",
        [1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female"
    )

    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=99,
        value=40
    )

    marital = st.selectbox(
        "Marital status",
        [1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            1: "Married",
            2: "Divorced",
            3: "Widowed",
            4: "Separated",
            5: "Never married",
            6: "In a relationship"
        }[x]
    )

    employ = st.selectbox(
        "Current employment status",
        [1, 2, 3, 4, 5, 6, 7, 8],
        format_func=lambda x: {
            1: "Employed for wages",
            2: "Self-employed",
            3: "Out of work for ≥ 1 year",
            4: "Out of work for < 1 year",
            5: "Homemaker",
            6: "Student",
            7: "Retired",
            8: "Unable to work"
        }[x]
    )
    st.markdown("<hr style='border:0.6px solid #E0E0E0'>", unsafe_allow_html=True)

    st.subheader("📏 Body Measurement")

    weight = st.number_input("Body weight (kg)", min_value=20.0, max_value=200.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=200.0)

    if height > 0:
        bmi = round(weight / ((height / 100) ** 2), 2)
    else:
        bmi = np.nan

    st.markdown("<hr style='border:0.6px solid #E0E0E0'>", unsafe_allow_html=True)

    st.subheader("🩺 General Health (Past 30 Days)")

    genhlth = st.selectbox(
        "How would you rate your general health?",
        [1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "Excellent",
            2: "Very good",
            3: "Good",
            4: "Fair",
            5: "Poor"
        }[x]
    )

    phys14 = st.selectbox(
        "Number of days your physical health was not good",
        [1, 2, 3],
        format_func=lambda x: {
            1: "0 days",
            2: "1–13 days",
            3: "14 days or more"
        }[x]
    )

    ment14 = st.selectbox(
        "Number of days your mental health was not good",
        [1, 2, 3],
        format_func=lambda x: {
            1: "0 days",
            2: "1–13 days",
            3: "14 days or more"
        }[x]
    )

    poorhlth = st.number_input(
        "Total days poor health limited your usual activities",
        min_value=0,
        max_value=30,
        value=0,
        help="Enter the number of days (0–30) when health problems limited daily activities"
    )

    st.markdown("<hr style='border:0.6px solid #E0E0E0'>", unsafe_allow_html=True)
    st.subheader("🏃 Lifestyle & Medical History")

    exercise = st.selectbox(
        "Did you do any physical activity or exercise in the past month?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    smoke = st.selectbox(
        "Have you smoked at least 100 cigarettes in your lifetime?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    heart = st.selectbox(
        "Have you ever been diagnosed with coronary heart disease?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    asthma = st.selectbox(
        "Asthma status",
        [1, 2, 3],
        format_func=lambda x: {
            1: "Current asthma (active diagnosis)",
            2: "Past history of asthma (not currently active)",
            3: "No history of asthma"
        }[x]
    )

    diabetes = st.selectbox(
        "Diabetes status",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Current diabetes (diagnosed)",
            2: "Gestational diabetes only (during pregnancy)",
            3: "No history of diabetes",
            4: "Pre-diabetes (at risk / borderline)"
        }[x]
    )

    diffwalk = st.selectbox(
        "Do you have serious difficulty walking or climbing stairs?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    arthritis = st.selectbox(
        "Have you ever been diagnosed with arthritis-related conditions?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    kidney = st.selectbox(
        "Have you ever been diagnosed with kidney disease?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    skin_cancer = st.selectbox(
        "History of non-melanoma skin cancer?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    other_cancer = st.selectbox(
        "History of melanoma or other types of cancer?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    depression = st.selectbox(
        "Have you ever been diagnosed with a depressive disorder?",
        [1, 2],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    submitted = st.form_submit_button("🔍 Predict Risk")

# =========================
# OUTPUT
# =========================
if submitted:

    user_df = pd.DataFrame([{
        "_SEX": sex,
        "_AGE80": age,
        "MARITAL": marital,
        "EMPLOY1": employ,
        "_BMI5": bmi * 100,
        "GENHLTH": genhlth,
        "_PHYS14D": phys14,
        "_MENT14D": ment14,
        "ADDEPEV3": depression,
        "POORHLTH": poorhlth,
        "EXERANY2": exercise,
        "SMOKE100": smoke,
        "CVDCRHD4": heart,
        "_ASTHMS1": asthma,
        "DIABETE4": diabetes,
        "DIFFWALK": diffwalk,
        "HAVARTH4": arthritis,
        "CHCKDNY2": kidney,
        "CHCSCNC1": skin_cancer,
        "CHCOCNC1": other_cancer
    }])

    prob = model.predict_proba(user_df)[0, 1]
    risk = probability_to_risk(prob)

    st.divider()
    
    st.markdown("""
    <div class="section-card">
        <h3>💡 Prediction Result</h3>
    </div>
    """, unsafe_allow_html=True)
    if risk <= 3:
        level = "Low Risk"
        color = "#2ECC71"
        recommendation = (
            "Your current profile indicates a lower level of risk. "
            "Continue maintaining a healthy lifestyle and perform regular skin self-checks."
        )
    elif risk <= 6:
        level = "Moderate Risk"
        color = "#F1C40F"
        recommendation = (
            "Some factors in your profile are associated with an increased risk."
            "Monitoring skin changes and considering preventive medical consultation is recommended."
        )
    else:
        level = "High Risk"
        color = "#E74C3C"
        recommendation = (
            "Several factors in your profile are associated with a higher risk level. "
            "A professional medical evaluation is strongly recommended for further assessment."
        )

# =========================
# Interpretation Text
# =========================
    st.markdown(f"""
    <h4>
    Risk Interpretation: 
    <span style="color:{color}; font-weight:600;">
    {level}
    </span>
    </h4>
    """, unsafe_allow_html=True)

    st.write(recommendation)

# =========================
# Gauge Visualization
# =========================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        number={'font': {'size': 48}},
        gauge={
            'axis': {'range': [1, 10], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [1, 3], 'color': '#EAF7EF'},
                {'range': [3, 6], 'color': '#FFF6DB'},
                {'range': [6, 10], 'color': '#FDEDEC'}
            ],
        }
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# Recommended Actions
# =========================
    st.markdown("""
    <div class="section-card">
        <h4>🩺 Recommended Actions</h4>
        <ul>
            <li>Consult a qualified healthcare professional or dermatologist</li>
            <li>Perform regular skin self-examinations</li>
            <li>Protect your skin from excessive sun exposure</li>
            <li>Seek medical advice if you notice unusual skin changes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



st.markdown("""
<p style="font-size:12px; color:#577C8E; text-align:center; margin-top:40px;">
⚠️ This tool is intended for educational and research purposes only.
It does not replace professional medical diagnosis or advice.
</p>
""", unsafe_allow_html=True)
