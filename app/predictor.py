"""Prediction helpers for the Streamlit app."""

import joblib
import pandas as pd

from .config import MODEL_PATH, PREPROCESSOR_PATH
from .utils import build_input_frame, clamp_prediction, get_feature_columns


def load_model_artifacts():
    """Load the saved model and preprocessing object from disk."""
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


def predict_admission(payload: dict) -> float | None:
    """Predict admission probability from a user payload."""
    model, preprocessor = load_model_artifacts()
    if model is None or preprocessor is None:
        return None

    input_df = build_input_frame(
        payload["gre_score"],
        payload["toefl_score"],
        payload["university_rating"],
        payload["sop"],
        payload["lor"],
        payload["cgpa"],
        payload["research"],
    )

    transformed = preprocessor.transform(input_df[get_feature_columns()])
    if hasattr(model, "predict_proba"):
        prediction = model.predict_proba(pd.DataFrame(transformed, columns=get_feature_columns()))[0, 1]
    else:
        prediction = float(model.predict(pd.DataFrame(transformed, columns=get_feature_columns()))[0])
    return clamp_prediction(prediction)
