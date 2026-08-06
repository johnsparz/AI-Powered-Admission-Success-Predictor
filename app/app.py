"""Main Streamlit app for admission prediction."""

import streamlit as st

from .config import ASSETS_DIR
from .explain import build_rule_based_explanation
from .predictor import predict_admission
from .utils import build_input_frame

st.set_page_config(
    page_title="Admission Success Predictor",
    page_icon="🎓",
    layout="centered",
)

st.title("AI-Powered Admission Success Predictor")
st.caption("Predict your chance of graduate admission from academic profile inputs.")

if (ASSETS_DIR / "logo.png").exists():
    st.image(str(ASSETS_DIR / "logo.png"), width=180)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        gre_score = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
        toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)
        university_rating = st.slider("University Rating", 1, 5, 3)
        research = st.selectbox("Research Experience", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
    with col2:
        sop = st.slider("Statement of Purpose (SOP)", 1.0, 5.0, 3.5, 0.5)
        lor = st.slider("Letter of Recommendation (LOR)", 1.0, 5.0, 3.5, 0.5)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.01)

    submitted = st.form_submit_button("Predict Admission Chance")

if submitted:
    payload = {
        "gre_score": gre_score,
        "toefl_score": toefl_score,
        "university_rating": university_rating,
        "sop": sop,
        "lor": lor,
        "cgpa": cgpa,
        "research": research,
    }
    prediction = predict_admission(payload)
    if prediction is None:
        st.warning("No trained model artifacts were found. Train the model first and place the files in the models folder.")
        st.stop()

    st.subheader("Prediction Result")
    st.metric("Chance of Admit", f"{prediction:.0%}")
    st.progress(prediction)
    st.success(build_rule_based_explanation(prediction))
