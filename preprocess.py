import re

# English + basic regional abusive words (extendable)
ABUSE_WORDS = [
    "fuck", "shit", "bitch", "ass", "idiot", "stupid", "bewakoof", "chu", "madarchod"
]

NEGATIVE_WORDS = [
    "worst", "useless", "bad", "boring", "confusing",
    "not good", "waste", "poor", "disappointed"
]

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\u0900-\u0D7F\s]", " ", text)  # supports Indian scripts
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_invalid(text: str) -> bool:
    if len(text.strip()) < 3:
        return True
    if re.fullmatch(r"[\d\s:.-]+", text):
        return True
    if re.fullmatch(r"[^\w]+", text):
        return True
    return False

def contains_abuse(text: str) -> bool:
    return any(word in text for word in ABUSE_WORDS)

def is_negative(text: str) -> bool:
    return any(word in text for word in NEGATIVE_WORDS)
