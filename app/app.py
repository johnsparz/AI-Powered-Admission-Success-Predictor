"""
Main Streamlit application.

AI-Powered Admission Success Predictor
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import APP_TITLE, PAGE_ICON
from app.predictor import predict_applicant
from app.explain import generate_explanation
from app.groq_explainer import generate_ai_explanation
from app.styles import load_css


def main() -> None:
    """
    Main Streamlit application.
    """

    # --------------------------------------------------------
    # Streamlit Configuration
    # --------------------------------------------------------

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_css()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.title("🎓 AI-Powered Admission Success Predictor")

    st.markdown(
        """
Predict graduate admission using a trained **Machine Learning**
(Logistic Regression) model.

The application also provides:

- 📊 Admission Probability
- 📋 Rule-Based Explanation
- 🤖 AI Explanation using Groq Llama 3.3
"""
    )
    
    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    with st.sidebar:

        st.header("📝 Applicant Information")

        st.markdown(
            """
Enter the applicant's academic profile below,
then click **Predict Admission**.
"""
        )

        gre_score = st.slider(
            "GRE Score",
            min_value=260,
            max_value=340,
            value=320,
        )

        toefl_score = st.slider(
            "TOEFL Score",
            min_value=80,
            max_value=120,
            value=105,
        )

        university_rating = st.selectbox(
            "University Rating",
            options=[1, 2, 3, 4, 5],
            index=2,
        )

        sop = st.slider(
            "Statement of Purpose (SOP)",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.5,
        )

        lor = st.slider(
            "Letter of Recommendation (LOR)",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.5,
        )

        cgpa = st.slider(
            "CGPA",
            min_value=6.0,
            max_value=10.0,
            value=8.5,
            step=0.1,
        )

        research = st.selectbox(
            "Research Experience",
            options=["No", "Yes"],
        )

        program = st.selectbox(
            "Program",
            options=[
                "Computer Science",
                "Data Science",
                "Artificial Intelligence",
                "Software Engineering",
                "Information Technology",
            ],
        )

        gender = st.selectbox(
            "Gender",
            options=[
                "Male",
                "Female",
            ],
        )

        application_completion = st.selectbox(
            "Application Status",
            options=[
                "Complete",
                "Incomplete",
            ],
        )

        predict_button = st.button(
            "🚀 Predict Admission",
            use_container_width=True,
        )
        
            # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if not predict_button:
        st.info(
            "Complete the applicant information in the sidebar and click **🚀 Predict Admission**."
        )
        return

    # Create applicant dataframe
    applicant = pd.DataFrame(
        {
            "gre_score": [gre_score],
            "toefl_score": [toefl_score],
            "university_rating": [university_rating],
            "sop": [sop],
            "lor": [lor],
            "cgpa": [cgpa],
            "research": [1 if research == "Yes" else 0],
            "program": [program],
            "application_completion": [application_completion],
            "gender": [gender],
        }
    )

    # Make prediction
    try:
        with st.spinner("Predicting admission outcome..."):
            prediction, probability = predict_applicant(applicant)

    except Exception as e:
        st.error(f"Prediction failed:\n\n{e}")
        st.stop()

    # Generate rule-based explanation
    reasons = generate_explanation(
        applicant.iloc[0].to_dict()
    )
    
        # --------------------------------------------------------
    # Display Prediction Result
    # --------------------------------------------------------

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.markdown(
            f"""
            <div class="prediction-success">
                <h2>🎉 Likely to be Admitted</h2>
                <h3>Admission Probability: {probability:.1%}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="prediction-fail">
                <h2>❌ Admission Unlikely</h2>
                <h3>Admission Probability: {probability:.1%}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Probability Metric
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Admission Probability",
            value=f"{probability:.1%}",
        )

    with col2:

        st.metric(
            label="Prediction",
            value="Admitted" if prediction == 1 else "Not Admitted",
        )

    # --------------------------------------------------------
    # Progress Bar
    # --------------------------------------------------------

    st.progress(float(probability))

    st.caption(
        f"The model estimates a **{probability:.1%}** chance of admission."
    )
        # --------------------------------------------------------
    # Explanations
    # --------------------------------------------------------

    st.divider()

    left_column, right_column = st.columns(2)

    # --------------------------------------------------------
    # Rule-Based Explanation
    # --------------------------------------------------------

    with left_column:

        st.subheader("📋 Rule-Based Explanation")

        for reason in reasons:
            st.success(reason)

    # --------------------------------------------------------
    # AI Explanation
    # --------------------------------------------------------

    with right_column:

        st.subheader("🤖 AI Explanation")

        with st.spinner("Generating AI explanation..."):

            try:

                ai_explanation = generate_ai_explanation(
                    applicant=applicant.iloc[0].to_dict(),
                    prediction=prediction,
                    probability=probability,
                    reasons=reasons,
                )

                st.write(ai_explanation)

            except Exception as e:

                st.error(
                    f"Unable to generate AI explanation.\n\n{e}"
                )
        # --------------------------------------------------------
    # Applicant Summary
    # --------------------------------------------------------

    st.divider()

    st.subheader("📄 Applicant Summary")

    st.dataframe(
        applicant,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Download CSV
    # --------------------------------------------------------

    csv = applicant.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Applicant Data",
        data=csv,
        file_name="applicant_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "AI-Powered Admission Success Predictor | "
        "Built with Streamlit, Scikit-learn, and Groq Llama 3.3"
    )


# ----------------------------------------------------------------------
# Application Entry Point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()