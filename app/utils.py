"""Helper utilities for form building and output formatting."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import FEATURE_COLUMNS  # noqa: E402


def build_input_frame(
    gre_score: int,
    toefl_score: int,
    university_rating: int,
    sop: float,
    lor: float,
    cgpa: float,
    research: int,
) -> pd.DataFrame:
    """Create a single-row dataframe matching the training feature schema."""
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


def clamp_prediction(value: float) -> float:
    """Keep predictions within the [0, 1] probability range."""
    return max(0.0, min(1.0, float(value)))


def get_feature_columns() -> list[str]:
    """Return the canonical feature columns used by preprocessing."""
    return FEATURE_COLUMNS
