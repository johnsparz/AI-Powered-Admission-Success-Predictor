"""
CSS styling for the Streamlit application.
"""

import streamlit as st


def load_css() -> None:
    """Load custom CSS into the Streamlit application."""

    st.markdown(
        """
        <style>

        /* Main Layout */
        .main > div {
            padding-top: 2rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
        }

        /* Success Card */
        .prediction-success {
            background-color: #ecfdf5;
            border-left: 6px solid #10b981;
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 20px;
        }

        /* Failure Card */
        .prediction-fail {
            background-color: #fef2f2;
            border-left: 6px solid #ef4444;
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 20px;
        }

        /* Section Headers */
        .section-title {
            font-size: 22px;
            font-weight: 700;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #1f2937;
        }

        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: white;
            border: 1px solid #e5e7eb;
            padding: 15px;
            border-radius: 10px;
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )