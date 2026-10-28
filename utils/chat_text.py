import re


ANSWER_LANGUAGE_LINE = re.compile(
    r"^\s*\**ANSWER LANGUAGE:[^\n]*\n*",
    re.IGNORECASE | re.MULTILINE,
)


def clean_answer(text):
    """Remove an echoed answer-language instruction from an assistant response."""
    return ANSWER_LANGUAGE_LINE.sub("", text or "").strip()
