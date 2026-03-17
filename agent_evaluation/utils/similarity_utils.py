import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_EMBEDDING_BACKEND = "sentence_transformers"

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            # Offline-safe fallback for evaluation environments where HuggingFace
            # downloads are blocked (proxy restrictions, no network, etc.).
            logger.warning(
                "SentenceTransformer unavailable; falling back to TF-IDF similarity. Error: %s",
                exc,
            )
            _model = None
            global _EMBEDDING_BACKEND
            _EMBEDDING_BACKEND = "tfidf"
    return _model

def compute_semantic_similarity(text1, text2):
    """
    Computes cosine similarity between two texts.

    Primary backend: SentenceTransformers (if available).
    Fallback backend: TF-IDF cosine similarity (offline-safe).
    """
    if not text1 or not text2:
        return 0.0

    model = get_model()
    if model is not None:
        embeddings = model.encode([text1, text2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(sim)

    # ---- TF-IDF fallback ----
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        mat = vec.fit_transform([text1, text2])
        sim = (mat[0] @ mat[1].T).A[0][0]
        return float(sim)
    except Exception:
        return 0.0
