"""Data loading and preprocessing utilities."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Admission_Predict.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]
TARGET_COLUMN = "Chance of Admit "


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw admission dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers and normalize column names."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    if "Serial No." in cleaned.columns:
        cleaned = cleaned.drop(columns=["Serial No."])
    return cleaned


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split cleaned data into features and target."""
    target_col = TARGET_COLUMN.strip()
    if target_col not in df.columns and "Chance of Admit" in df.columns:
        target_col = "Chance of Admit"
    X = df[FEATURE_COLUMNS]
    y = df[target_col]
    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize feature columns using training statistics."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def prepare_datasets(
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Load, clean, split, and scale the dataset."""
    df = clean_data(load_raw_data())
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
    }


def save_processed_data(datasets: dict, output_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Persist processed train/test splits to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets["X_train"].to_csv(output_dir / "X_train.csv", index=False)
    datasets["X_test"].to_csv(output_dir / "X_test.csv", index=False)
    datasets["y_train"].to_csv(output_dir / "y_train.csv", index=False)
    datasets["y_test"].to_csv(output_dir / "y_test.csv", index=False)
