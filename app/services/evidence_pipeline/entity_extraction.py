import os
import logging
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy model
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        except ImportError:
            logger.warning("spaCy not installed. Run: pip install spacy")
    return _nlp


def extract_entities(text: str) -> list:
    """
    Extract named entities from text using spaCy NER.
    Filters to relevant entity types: PERSON, ORG, GPE, EVENT, LOC.

    Returns a deduplicated list of entity strings.
    """
    nlp = get_nlp()
    if not nlp:
        return []

    try:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT", "LOC"}:
                entities.append(ent.text)
        return list(set(entities))
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return []
