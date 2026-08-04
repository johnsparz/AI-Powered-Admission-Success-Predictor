"""Model evaluation utilities."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import prepare_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_model(model_path: Path | None = None):
    """Load the latest trained model."""
    if model_path is None:
        model_path = MODELS_DIR / "admission_model.joblib"
    return joblib.load(model_path)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute common regression metrics."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_model(model, datasets: dict) -> dict[str, float]:
    """Evaluate a trained model on the held-out test set."""
    predictions = model.predict(datasets["X_test"])
    return regression_metrics(datasets["y_test"], predictions)


def save_metrics(metrics: dict[str, float]) -> Path:
    """Write evaluation metrics to the reports directory."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics_path


def plot_predictions(y_true, y_pred, output_path: Path) -> Path:
    """Create a predicted vs actual scatter plot."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Actual Chance of Admit")
    plt.ylabel("Predicted Chance of Admit")
    plt.title("Predicted vs Actual Admission Success")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main() -> None:
    datasets = prepare_datasets()
    model = load_model()
    metrics = evaluate_model(model, datasets)
    metrics_path = save_metrics(metrics)
    plot_path = plot_predictions(
        datasets["y_test"],
        model.predict(datasets["X_test"]),
        REPORTS_DIR / "predicted_vs_actual.png",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
