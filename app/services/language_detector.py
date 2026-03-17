from langdetect import detect

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns ISO language code.
    """
    try:
        return detect(text)
    except:
        return "unknown"
