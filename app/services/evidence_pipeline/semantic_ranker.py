import os
import logging
import warnings
import datetime

logger = logging.getLogger(__name__)

# Suppress verbose HuggingFace/SentenceTransformer output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.models.model_loader import get_embedding_model


def get_model():
    return get_embedding_model()


def compute_recency_score(timestamp_str: str | None) -> float:
    """
    Returns a 0.0–1.0 score: 1.0 for very recent evidence, decaying with age.
    Evidence older than 2 years gets 0.0. Missing timestamps return 0.5 (neutral).
    """
    if not timestamp_str:
        return 0.5
    try:
        ts = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        age_days = (now - ts).days
        if age_days <= 7:
            return 1.0
        elif age_days <= 30:
            return 0.85
        elif age_days <= 180:
            return 0.70
        elif age_days <= 365:
            return 0.50
        elif age_days <= 730:
            return 0.25
        return 0.0
    except Exception:
        return 0.5


def rank_evidence(claim: str, evidence_items: list, use_nli: bool = False) -> list:
    """
    Rank a list of evidence items against a claim.

    Each item in evidence_items should be a dict with at least:
        {"text": "...", "trust_score": 0.0-1.0}

    IMPORTANT:
    - Ranking MUST be driven purely by semantic similarity.
    - Source trust and recency are surfaced as metadata only and are
      consumed later by the credibility engine, not by this ranker.
    """
    if not evidence_items or not claim:
        return []

    model = get_model()
    if not model:
        return []

    try:
        texts = [item["text"] for item in evidence_items]
        claim_embedding = model.encode([claim])
        texts_embeddings = model.encode(texts)
        similarities = cosine_similarity(claim_embedding, texts_embeddings)[0]

        ranked_results = []
        for idx, sim_score in enumerate(similarities):
            sem_score = float(sim_score)
            trust_score = float(evidence_items[idx].get("trust_score", 0.5))
            timestamp = evidence_items[idx].get("published_at") or evidence_items[idx].get("timestamp")

            # NOTE: We intentionally do NOT blend trust / recency into the
            # ranking score. Those dimensions are handled later during
            # credibility scoring. Here we keep:
            #   score == semantic similarity
            composite_score = sem_score

            result_item = {
                "score": composite_score,
                "semantic_score": sem_score,
                "trust_score": trust_score,
                "recency_score": compute_recency_score(timestamp),
                "text": texts[idx],
                "source": evidence_items[idx].get("source", "Unknown"),
                "domain": evidence_items[idx].get("domain", "unknown"),
                "timestamp": timestamp,
            }

            ranked_results.append(result_item)

        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        return ranked_results

    except Exception:
        logger.exception("Error in semantic ranking")
        return []
