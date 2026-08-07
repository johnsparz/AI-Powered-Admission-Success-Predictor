"""
Rule-based explanation module.

This module provides simple, human-readable explanations
based on the applicant's profile.
"""

from typing import Dict, List


def generate_explanation(applicant: Dict) -> List[str]:
    """
    Generate rule-based explanations for the applicant.

    Parameters
    ----------
    applicant : dict
        Applicant information.

    Returns
    -------
    list[str]
        List of explanation strings.
    """

    reasons = []

    # GRE Score
    if applicant["gre_score"] >= 325:
        reasons.append("Strong GRE score strengthens the application.")
    elif applicant["gre_score"] >= 315:
        reasons.append("GRE score is competitive.")
    else:
        reasons.append("Improving the GRE score may increase admission chances.")

    # TOEFL Score
    if applicant["toefl_score"] >= 110:
        reasons.append("Excellent TOEFL score demonstrates strong English proficiency.")
    elif applicant["toefl_score"] >= 100:
        reasons.append("TOEFL score meets the expectations of many universities.")
    else:
        reasons.append("A higher TOEFL score could strengthen the application.")

    # CGPA
    if applicant["cgpa"] >= 9.0:
        reasons.append("Outstanding CGPA is one of the strongest aspects of the application.")
    elif applicant["cgpa"] >= 8.0:
        reasons.append("CGPA is competitive for graduate admission.")
    else:
        reasons.append("Improving academic performance would increase competitiveness.")

    # University Rating
    if applicant["university_rating"] >= 4:
        reasons.append("Graduating from a highly rated university is an advantage.")

    # SOP
    if applicant["sop"] >= 4:
        reasons.append("A strong Statement of Purpose supports the application.")

    # LOR
    if applicant["lor"] >= 4:
        reasons.append("Strong Letters of Recommendation add credibility.")

    # Research
    if applicant["research"] == 1:
        reasons.append("Research experience significantly strengthens the profile.")
    else:
        reasons.append("Research experience could improve the application.")

    # Application Status
    if applicant["application_completion"] == "Complete":
        reasons.append("Application is complete and ready for evaluation.")
    else:
        reasons.append("Complete the application before submission.")

    return reasons


if __name__ == "__main__":

    sample = {
        "gre_score": 328,
        "toefl_score": 112,
        "university_rating": 5,
        "sop": 4.5,
        "lor": 4.5,
        "cgpa": 9.3,
        "research": 1,
        "application_completion": "Complete",
    }

    print(generate_explanation(sample))