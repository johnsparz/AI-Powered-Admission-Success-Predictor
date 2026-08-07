import pandas as pd

from app.config import MODEL_PATH, PREPROCESSOR_PATH

sample = pd.DataFrame({
    "gre_score": [325],
    "toefl_score": [110],
    "university_rating": [4],
    "sop": [4.0],
    "lor": [4.0],
    "cgpa": [9.0],
    "research": [1],
    "program": ["Computer Science"],
    "application_completion": ["Complete"],
    "gender": ["Male"]
})

prediction, probability = predict_applicant(sample)

print("Prediction:", prediction)
print("Probability:", probability)