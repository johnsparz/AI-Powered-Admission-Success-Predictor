"""Model training pipeline."""

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from preprocessing import prepare_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def train_baseline_model(datasets: dict) -> LinearRegression:
    """Train a simple linear regression baseline."""
    model = LinearRegression()
    model.fit(datasets["X_train"], datasets["y_train"])
    return model


def train_random_forest(datasets: dict) -> RandomForestRegressor:
    """Train a random forest regressor."""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(datasets["X_train"], datasets["y_train"])
    return model


def save_model(model, name: str = "admission_model") -> Path:
    """Persist a trained model to the models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)
    return model_path


def save_scaler(scaler, name: str = "feature_scaler") -> Path:
    """Persist the fitted feature scaler."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(scaler, scaler_path)
    return scaler_path


def save_training_metadata(datasets: dict, model_name: str) -> Path:
    """Save metadata required for inference and evaluation."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "feature_columns": datasets["feature_columns"],
        "train_size": len(datasets["X_train"]),
        "test_size": len(datasets["X_test"]),
    }
    metadata_path = REPORTS_DIR / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def main() -> None:
    datasets = prepare_datasets()
    model = train_random_forest(datasets)
    model_path = save_model(model)
    scaler_path = save_scaler(datasets["scaler"])
    metadata_path = save_training_metadata(datasets, model_path.stem)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
