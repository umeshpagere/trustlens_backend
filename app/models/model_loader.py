import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Offline-safe fallback embedder (for eval / restricted environments)
# ---------------------------------------------------------------------------
class _TfidfEmbedder:
    """
    Minimal encode()-compatible embedder used when SentenceTransformer
    cannot be loaded (e.g., HuggingFace downloads blocked).

    Note: This is NOT intended for production-quality retrieval; it exists
    so the pipeline and evaluation harness can run deterministically offline.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        self._np = np
        self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
        self._fitted = False

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [t or "" for t in texts]

        # Fit on-the-fly per batch (keeps runtime stable for small eval sets).
        # This means vectors are only comparable within the batch, but our
        # ranker always embeds claim + candidates together, so it's acceptable.
        mat = self._vec.fit_transform(texts)
        arr = mat.toarray().astype("float32")
        # L2 normalize
        norms = self._np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms


# ---------------------------------------------------------------------------
# ML Model Preloading
# ---------------------------------------------------------------------------
print("⚙️ [Startup] Preloading ML models...")
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully")
    print("✅ Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    try:
        embedding_model = _TfidfEmbedder()
        logger.warning("Using TF-IDF embedder fallback for embeddings.")
        print("⚠️ Using TF-IDF embedder fallback (offline-safe).")
    except Exception as tfidf_err:
        logger.error(f"Failed to initialize TF-IDF fallback embedder: {tfidf_err}")
        embedding_model = None

def get_embedding_model():
    """Return the preloaded SentenceTransformer model.

    Raises RuntimeError if the model failed to load at startup,
    so callers fail loudly rather than silently producing empty embeddings.
    """
    if embedding_model is None:
        raise RuntimeError(
            "Embedding model (all-MiniLM-L6-v2) failed to load at startup. "
            "Check model_cache and Dockerfile pre_download_models.py."
        )
    return embedding_model

