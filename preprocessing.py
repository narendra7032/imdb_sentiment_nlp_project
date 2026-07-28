"""
Text preprocessing utilities for IMDB sentiment analysis.
"""
import re

HTML_TAG_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, remove punctuation/numbers, collapse whitespace."""
    text = str(text).lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text
