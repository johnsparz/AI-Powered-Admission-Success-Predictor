"""Feature engineering helpers for admission prediction."""

import pandas as pd

from preprocessing import FEATURE_COLUMNS


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple interaction features from exam scores and academic metrics."""
    engineered = df.copy()
    engineered["GRE_TOEFL_ratio"] = engineered["GRE Score"] / engineered["TOEFL Score"]
    engineered["Academic_strength"] = (
        engineered["CGPA"] * engineered["University Rating"]
    )
    engineered["Recommendation_score"] = engineered["SOP"] + engineered["LOR"]
    return engineered


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the base feature set used by the model."""
    return df[FEATURE_COLUMNS]
