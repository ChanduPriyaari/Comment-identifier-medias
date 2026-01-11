import re
import string

def clean_text(text: str) -> str:
    """
    Clean raw text while preserving semantic meaning.
    """

    # Lowercase for consistency
    text = text.lower()

    # Replace URLs
    text = re.sub(r"http\S+|www\S+", "<URL>", text)

    # Replace user mentions
    text = re.sub(r"@\w+", "<USER>", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove punctuation (keep words intact)
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text
