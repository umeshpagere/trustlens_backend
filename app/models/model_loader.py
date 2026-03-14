import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ML Model Preloading
# ---------------------------------------------------------------------------
print("⚙️ [Startup] Preloading ML models...")
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully")
    print("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None

def get_embedding_model():
    """Access the preloaded model."""
    return embedding_model
