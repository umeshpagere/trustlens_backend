import os
import logging
import warnings

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
from .nli_verifier import check_contradiction, compute_nli_score
from app.models.model_loader import load_embedding_model

def get_model():
    return load_embedding_model()


def rank_evidence(claim: str, evidence_items: list, use_nli: bool = False) -> list:
    """
    Rank a list of evidence items against a claim.
    Each item in evidence_items should be a dict: {"text": "...", "trust_score": 0.0-1.0}

    The final ranking score is a composite:
        composite_score = (semantic_similarity * 0.7) + (source_trust * 0.3)

    If use_nli=True, NLI score is also considered (advanced mode).
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

            # Composite calculation: 70% semantic, 30% trust
            composite_score = (sem_score * 0.7) + (trust_score * 0.3)

            result_item = {
                "score": composite_score,
                "semantic_score": sem_score,
                "trust_score": trust_score,
                "text": texts[idx],
                "source": evidence_items[idx].get("source", "Unknown")
            }

            if use_nli:
                nli_result = check_contradiction(claim, texts[idx])
                nli_contribution = compute_nli_score(nli_result)
                # If using NLI, we blend it into the composite score
                result_item["nli_label"] = nli_result["label"]
                result_item["nli_score"] = nli_result["score"]
                # Adjust composite with NLI: Replace semantic portion with NLI-blended semantic
                blended_sem = (sem_score * 0.6) + (nli_contribution * 0.4)
                result_item["score"] = (blended_sem * 0.7) + (trust_score * 0.3)

            ranked_results.append(result_item)

        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        return ranked_results

    except Exception as e:
        logger.error(f"Error in semantic ranking: {e}")
        return []

    except Exception as e:
        logger.error(f"Error in semantic ranking: {e}")
        return []
