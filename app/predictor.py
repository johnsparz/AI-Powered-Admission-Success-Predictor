"""
Prediction module for the AI-Powered Admission Success Predictor.
"""

from __future__ import annotations

import joblib
import pandas as pd

from app.config import MODEL_PATH, PREPROCESSOR_PATH

# ---------------------------------------------------------------------
# Load the model and preprocessor once
# ---------------------------------------------------------------------

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"Failed to load model:\n{e}"
    )

try:
    PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    raise RuntimeError(
        f"Failed to load preprocessor:\n{e}"
    )


# ---------------------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------------------

def predict_applicant(applicant: pd.DataFrame) -> tuple[int, float]:
    """
    Predict admission for a single applicant.

    Parameters
    ----------
    applicant : pd.DataFrame
        Single-row dataframe containing applicant details.

    Returns
    -------
    tuple[int, float]

    prediction:
        0 = Not Admitted
        1 = Admitted

    probability:
        Admission probability.
    """

    if applicant.empty:
        raise ValueError("Applicant dataframe is empty.")

    # Transform the input
    transformed = PREPROCESSOR.transform(applicant)

    # Predict class
    prediction = int(MODEL.predict(transformed)[0])

    # Predict probability
    probability = float(
        MODEL.predict_proba(transformed)[0][1]
    )

    return prediction, probability


# ---------------------------------------------------------------------
# Optional CLI test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    sample = pd.DataFrame(
        {
            "gre_score": [325],
            "toefl_score": [110],
            "university_rating": [4],
            "sop": [4.0],
            "lor": [4.0],
            "cgpa": [9.0],
            "research": [1],
            "program": ["Computer Science"],
            "application_completion": ["Complete"],
            "gender": ["Male"],
        }
    )

    prediction, probability = predict_applicant(sample)

    print("-" * 40)
    print("Prediction :", prediction)
    print("Probability:", round(probability, 4))
    print("-" * 40)