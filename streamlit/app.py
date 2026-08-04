"""Streamlit app for graduate admission success prediction."""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import FEATURE_COLUMNS  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "admission_model.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "feature_scaler.joblib"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def build_input_frame(
    gre_score: int,
    toefl_score: int,
    university_rating: int,
    sop: float,
    lor: float,
    cgpa: float,
    research: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GRE Score": gre_score,
                "TOEFL Score": toefl_score,
                "University Rating": university_rating,
                "SOP": sop,
                "LOR": lor,
                "CGPA": cgpa,
                "Research": research,
            }
        ]
    )


st.set_page_config(
    page_title="Admission Success Predictor",
    page_icon="🎓",
    layout="centered",
)

st.title("AI-Powered Admission Success Predictor")
st.caption("Predict your chance of graduate admission from academic profile inputs.")

model, scaler = load_artifacts()
if model is None or scaler is None:
    st.warning(
        "No trained model found. Run `python src/train_model.py` from the project root first."
    )
    st.stop()

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
    input_df = build_input_frame(
        gre_score,
        toefl_score,
        university_rating,
        sop,
        lor,
        cgpa,
        research,
    )
    X_scaled = pd.DataFrame(
        scaler.transform(input_df[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS,
    )
    prediction = float(model.predict(X_scaled)[0])
    prediction = max(0.0, min(1.0, prediction))

    st.subheader("Prediction Result")
    st.metric("Chance of Admit", f"{prediction:.0%}")
    st.progress(prediction)

    if prediction >= 0.75:
        st.success("Strong profile — high likelihood of admission.")
    elif prediction >= 0.5:
        st.info("Moderate profile — competitive but improvable.")
    else:
        st.warning("Lower predicted chance — consider strengthening SOP, LOR, or test scores.")
