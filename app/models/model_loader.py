import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Global model instance
_model = None

def load_embedding_model():
    """
    Loads the sentence-transformer model once and caches it.
    Used for semantic ranking and evidence alignment.
    """
    global _model

    if _model is None:
        logger.info("Loading sentence transformer model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")

    return _model

def get_model():
    """Access the preloaded model."""
    return _model
