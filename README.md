# AI-Powered Admission Success Predictor

An end-to-end machine learning project that predicts graduate admission success using Scikit-learn, SHAP, and Streamlit.

## Project Structure

```
AI-Powered-Admission-Success-Predictor/
├── data/
│   ├── raw/                  # Raw dataset
│   └── processed/            # Processed train/test splits
├── notebooks/                # Exploratory analysis
├── src/                      # Training and evaluation pipeline
├── streamlit/                # Interactive prediction app
├── models/                   # Saved model artifacts
├── reports/                  # Metrics and explainability plots
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow

1. **Explore data** — open `notebooks/AI_Admission_Predictor.ipynb`
2. **Train model** — `python src/train_model.py`
3. **Evaluate** — `python src/evaluate_model.py`
4. **Explain predictions** — `python src/explainability.py`
5. **Launch app** — `streamlit run streamlit/app.py`

## Dataset

Place the full `Admission_Predict.csv` file in `data/raw/`. A sample file is included for development.

## License

See [LICENSE](LICENSE).
