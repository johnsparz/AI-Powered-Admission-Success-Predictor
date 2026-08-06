"""Optional AI explanation integration for the app."""

from typing import Optional


def build_llm_explanation(prediction: float, text: str) -> Optional[str]:
    """Placeholder for future Groq/OpenAI integration."""
    if not text:
        return None
    return (
        f"AI summary: your predicted admission probability is {prediction:.0%}. "
        f"The current profile suggests {text.lower()}."
    )
