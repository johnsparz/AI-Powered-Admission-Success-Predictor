from app.explain import generate_explanation
from app.groq_explainer import generate_ai_explanation

candidate = {
    "gre_score": 330,
    "toefl_score": 114,
    "university_rating": 5,
    "sop": 4.5,
    "lor": 4.5,
    "cgpa": 9.4,
    "research": 1,
    "program": "Computer Science",
    "gender": "Male",
    "application_completion": "Complete"
}

prediction = 1
probability = 0.94

reasons = generate_explanation(candidate)

response = generate_ai_explanation(
    applicant=candidate,
    prediction=prediction,
    probability=probability,
    reasons=reasons
)

print(response)