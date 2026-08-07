import pandas as pd

from app.predictor import predict_applicant

candidate = pd.DataFrame({
    "gre_score": [330],
    "toefl_score": [114],
    "university_rating": [5],
    "sop": [4.5],
    "lor": [4.5],
    "cgpa": [9.4],
    "research": [1],
    "program": ["Computer Science"],
    "application_completion": ["Complete"],
    "gender": ["Male"]
})

prediction, probability = predict_applicant(candidate)

print(prediction)
print(probability)