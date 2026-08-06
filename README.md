# AI-Powered Admission Success Predictor

An end-to-end machine learning project that predicts graduate admission success using Scikit-learn, SHAP, and Streamlit.

## Project Structure

```
AI-Powered-Admission-Predictor/
├── app/
│   ├── app.py               # Main Streamlit app
│   ├── predictor.py         # Model prediction logic
│   ├── explain.py           # Rule-based explanations
│   ├── llm_explainer.py     # AI explanation (Groq/OpenAI)
│   ├── config.py            # Configuration
│   └── utils.py             # Helper functions
├── models/
│   ├── best_model.pkl
│   └── preprocessor.pkl
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── AI_Admission_Predictor.ipynb
├── reports/
│   └── logistic_coefficients.csv
├── assets/
│   ├── logo.png
│   └── banner.png
├── requirements.txt
├── README.md
└── LICENSE
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
5. **Launch app** — `streamlit run app/app.py` or `streamlit run streamlit/app.py`

## Dataset

Place the full `Admission_Predict.csv` file in `data/raw/`. A sample file is included for development.

## License

See [LICENSE](LICENSE).
