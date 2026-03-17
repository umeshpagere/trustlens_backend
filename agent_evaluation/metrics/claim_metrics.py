import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Lazy-loaded model to avoid overhead if not needed immediately
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model: all-MiniLM-L6-v2")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def evaluate_claim_extraction(expected_claims, predicted_claims, threshold=0.75):
    """
    Compares predicted claims with expected claims using semantic similarity.
    
    Args:
        expected_claims (list[str]): Ground truth claims.
        predicted_claims (list[str]): Claims extracted by the agent.
        threshold (float): Similarity threshold for a match.
        
    Returns:
        dict: {
            "claim_similarity_score": float (average max similarity),
            "matches": list[dict],
            "accuracy": float (proportion of expected claims found)
        }
    """
    if not expected_claims:
        return {"claim_similarity_score": 0.0, "matches": [], "accuracy": 0.0}
    if not predicted_claims:
        return {"claim_similarity_score": 0.0, "matches": [], "accuracy": 0.0}

    model = get_model()
    expected_embeddings = model.encode(expected_claims)
    predicted_embeddings = model.encode(predicted_claims)

    similarities = cosine_similarity(expected_embeddings, predicted_embeddings)
    
    max_similarities = np.max(similarities, axis=1)
    avg_max_similarity = np.mean(max_similarities)
    
    matches_found = np.sum(max_similarities >= threshold)
    accuracy = matches_found / len(expected_claims)

    return {
        "claim_similarity_score": float(avg_max_similarity),
        "claim_extraction_correct": accuracy >= 1.0, # True if all GT claims are matched
        "accuracy": float(accuracy)
    }

def evaluate_query_relevance(claim, queries):
    """
    Evaluates the relevance of generated queries to the primary claim.
    """
    if not queries or not claim:
        return {"query_relevance_score": 0.0}

    model = get_model()
    claim_embedding = model.encode([claim])
    query_embeddings = model.encode(queries)

    similarities = cosine_similarity(claim_embedding, query_embeddings)[0]
    max_relevance = np.max(similarities)

    return {
        "query_relevance_score": float(max_relevance)
    }
