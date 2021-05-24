import re


def clean_text(text: str, to_lower: bool = False) -> str:
    cleaned_text = re.sub('[^A-Za-z0-9]+', ' ', str(text))
    if to_lower:
        return cleaned_text.lower()
    return cleaned_text
