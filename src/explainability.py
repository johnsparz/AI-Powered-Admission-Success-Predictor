"""SHAP-based model explainability."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from preprocessing import prepare_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_model(model_path: Path | None = None):
    """Load the trained admission model."""
    if model_path is None:
        model_path = MODELS_DIR / "admission_model.joblib"
    return joblib.load(model_path)


def build_explainer(model, X_train: pd.DataFrame):
    """Create a SHAP explainer for tree-based models."""
    explainer = shap.TreeExplainer(model)
    return explainer, explainer.shap_values(X_train)


def save_summary_plot(shap_values, X_train: pd.DataFrame, output_path: Path) -> Path:
    """Save a SHAP summary plot to disk."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def explain_instance(
    model,
    instance: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return SHAP values for a single prediction instance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_values[0],
        }
    ).sort_values("shap_value", key=abs, ascending=False)


def main() -> None:
    datasets = prepare_datasets()
    model = load_model()
    shap_values = build_explainer(model, datasets["X_train"])[1]
    output_path = save_summary_plot(
        shap_values,
        datasets["X_train"],
        REPORTS_DIR / "shap_summary.png",
    )
    print(f"SHAP summary saved to {output_path}")


if __name__ == "__main__":
    main()
