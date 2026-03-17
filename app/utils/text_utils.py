import re

def normalize_claim_text(text: str) -> str:
    """
    Normalize claim text for deduplication while preserving semantic meaning.
    """

    if not text:
        return ""

    text = text.strip()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # preserve numeric indicators such as %, ., -
    text = re.sub(r"[^\w\s.%\-]", "", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()
