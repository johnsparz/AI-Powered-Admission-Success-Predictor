# 🎓 AI-Powered Admission Success Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Groq](https://img.shields.io/badge/Groq-Llama3.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **🚀 Live Application:** https://ai-powered-admission-success-predictor.streamlit.app/

Predict a graduate applicant's likelihood of admission using a Machine Learning model and receive both rule-based and AI-powered explanations generated with **Groq Llama 3.3**.

---

## 🌐 Live Demo

### 🚀 Streamlit Application

**Click here to use the application:**

### https://ai-powered-admission-success-predictor.streamlit.app/

---

## 📂 GitHub Repository

> Replace with your repository link after publishing.

https://github.com/johnsparz/AI-Powered-Admission-Success-Predictor

# 📊 Dataset

This project uses the **Graduate Admission Dataset** obtained from Kaggle.

The dataset contains academic and application-related information commonly used to estimate a student's likelihood of graduate admission.

## Original Dataset

**Source:**

https://www.kaggle.com/datasets/mohansacharya/graduate-admissions

---

## Features

| Feature | Description |
|----------|-------------|
| GRE Score | Graduate Record Examination score |
| TOEFL Score | English language proficiency score |
| University Rating | Rating of undergraduate institution |
| SOP | Statement of Purpose strength |
| LOR | Letter of Recommendation strength |
| CGPA | Undergraduate GPA |
| Research | Research experience (Yes/No) |
| Program | Intended graduate program *(engineered feature)* |
| Gender | Applicant gender *(engineered feature)* |
| Application Completion | Application status *(engineered feature)* |
| Chance of Admit | Target variable |

---

## Dataset Cleaning

The original dataset was cleaned before model training.

Cleaning steps included:

- Removed duplicate records
- Verified missing values
- Corrected data types
- Standardized feature names
- Added engineered categorical features for demonstration purposes:
  - Program
  - Gender
  - Application Completion
- Prepared categorical variables for one-hot encoding
- Standardized numerical variables using Scikit-learn pipelines

The cleaned dataset is included in the repository under the **data/** directory together with instructions for reproducing the preprocessing pipeline.

---

# 🤖 Machine Learning Pipeline

The project follows a complete end-to-end Machine Learning workflow.

## 1. Data Collection

- Graduate Admission Dataset (Kaggle)

---

## 2. Data Cleaning

Performed using **Pandas**.

Tasks included:

- Missing value inspection
- Duplicate removal
- Data validation
- Feature standardization
- Feature engineering

---

## 3. Exploratory Data Analysis (EDA)

Performed using:

- Pandas
- Matplotlib
- Plotly

Visualizations include:

- Distribution plots
- Correlation heatmap
- Feature relationships
- Admission probability distribution
- Research vs Admission
- CGPA vs Admission
- GRE vs Admission

---

## 4. Feature Engineering

Categorical variables:

- OneHotEncoder

Numerical variables:

- StandardScaler

Combined using:

- ColumnTransformer

---

## 5. Model Training

The following models were evaluated:

- Logistic Regression ✅
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors

The best-performing model (Logistic Regression) was selected based on evaluation metrics.

---

## 6. Model Evaluation

The trained models were evaluated using multiple classification metrics to ensure robust performance and generalization.

**Evaluation metrics included:**

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

After evaluation, the best-performing model (**Logistic Regression**) and the preprocessing pipeline were serialized using **Joblib** for deployment.

```text
models/
├── best_model.pkl
└── preprocessor.pkl
```

---

# 📒 Jupyter Notebook

The repository includes a fully documented **Jupyter Notebook** that demonstrates the complete end-to-end Machine Learning workflow.

The notebook covers:

- Data loading
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Data preprocessing
- Model training
- Model comparison
- Hyperparameter tuning
- Model evaluation
- Data visualizations
- Model serialization

For easier review without requiring Jupyter Notebook, both **HTML** and **PDF** exports of the notebook are also included in the repository.

# 📈 Visualizations

The project includes multiple visualizations created using:

- Matplotlib
- Pandas
- Plotly

Visualizations include:

- Correlation Matrix
- Feature Distributions
- Admission Probability Distribution
- CGPA Analysis
- GRE Score Analysis
- TOEFL Score Analysis
- Research Impact
- Model Performance Charts

These visualizations are available in both the Jupyter Notebook and the generated reports.

# ⚙️ Installation

Clone the repository.

```bash
git clone https://github.com/johnsparz/AI-Powered-Admission-Success-Predictor.git

cd AI-Powered-Admission-Success-Predictor
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file.

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Run the application.

```bash
streamlit run streamlit_app.py
```
