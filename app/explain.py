"""Rule-based explanation helpers."""


def build_rule_based_explanation(prediction: float) -> str:
    """Return a plain-language explanation for a prediction."""
    if prediction >= 0.75:
        return "Strong profile — your academic indicators suggest a high likelihood of admission."
    if prediction >= 0.5:
        return "Moderate profile — your results are competitive, though some areas can be strengthened."
    return "Lower projected chance — improving your test scores or supporting materials may improve your outcome."
