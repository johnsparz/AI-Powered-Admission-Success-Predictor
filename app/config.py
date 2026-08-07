"""
Configuration module for the AI-Powered Admission Success Predictor.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

APP_DIR = BASE_DIR / "app"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# ---------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------

MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# ---------------------------------------------------------------------
# Report files
# ---------------------------------------------------------------------

COEFFICIENTS_PATH = REPORTS_DIR / "logistic_coefficients.csv"

# ---------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------

APP_TITLE = "AI-Powered Admission Success Predictor"
PAGE_ICON = "🎓"

# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessor file not found:\n{PREPROCESSOR_PATH}"
    )

if GROQ_API_KEY is None:
    print("Warning: GROQ_API_KEY not found. AI explanations will not work.")