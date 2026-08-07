"""
Groq AI explanation module.
"""

from __future__ import annotations

from groq import Groq

from app.config import GROQ_API_KEY


# ---------------------------------------------------------------------
# Initialize Groq Client
# ---------------------------------------------------------------------

if not GROQ_API_KEY:
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------
# AI Explanation Function
# ---------------------------------------------------------------------

def generate_ai_explanation(
    applicant: dict,
    prediction: int,
    probability: float,
    reasons: list[str],
) -> str:
    """
    Generate an AI explanation using Groq.

    Parameters
    ----------
    applicant : dict
        Applicant information.

    prediction : int
        0 = Not Admitted
        1 = Admitted

    probability : float
        Admission probability.

    reasons : list[str]
        Rule-based explanation list.

    Returns
    -------
    str
        AI-generated explanation.
    """

    if client is None:
        return (
            "Groq API key not found. "
            "Please add GROQ_API_KEY to your .env file."
        )

    admission_status = (
        "Likely Admitted"
        if prediction == 1
        else "Admission Unlikely"
    )

    rule_text = "\n".join(
        f"- {reason}" for reason in reasons
    )

    prompt = f"""
You are an experienced university admissions advisor.

A machine learning model has already made a prediction.

Do NOT change the prediction.

Applicant Information

GRE Score: {applicant['gre_score']}
TOEFL Score: {applicant['toefl_score']}
CGPA: {applicant['cgpa']}
University Rating: {applicant['university_rating']}
SOP: {applicant['sop']}
LOR: {applicant['lor']}
Research: {"Yes" if applicant['research'] == 1 else "No"}
Program: {applicant['program']}
Gender: {applicant['gender']}
Application Status: {applicant['application_completion']}

Prediction:
{admission_status}

Admission Probability:
{probability:.2%}

Rule-Based Analysis:

{rule_text}

Write a concise explanation (120–180 words) that:

- Explains the prediction.
- Highlights strengths.
- Mentions weaknesses if any.
- Suggests realistic improvements.
- Uses professional and encouraging language.
- Do not contradict the prediction.
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.3,
            max_tokens=300,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Unable to generate AI explanation.\n\n{e}"


# ---------------------------------------------------------------------
# Local Test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    sample = {
        "gre_score": 330,
        "toefl_score": 114,
        "cgpa": 9.4,
        "university_rating": 5,
        "sop": 4.5,
        "lor": 4.5,
        "research": 1,
        "program": "Computer Science",
        "gender": "Male",
        "application_completion": "Complete",
    }

    reasons = [
        "Excellent CGPA.",
        "Strong GRE score.",
        "Research experience.",
    ]

    print(
        generate_ai_explanation(
            applicant=sample,
            prediction=1,
            probability=0.94,
            reasons=reasons,
        )
    )