import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Lazy-loaded model
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def evaluate_retrieval(claim, evidence_texts, threshold=0.6):
    """
    Evaluates evidence relevance against a claim.
    
    Args:
        claim (str): The claim being verified.
        evidence_texts (list[str]): Retrieved evidence snippets.
        threshold (float): Similarity threshold for relevance.
        
    Returns:
        dict: {
            "retrieval_score": float (max similarity),
            "relevant_evidence_count": int
        }
    """
    if not evidence_texts or not claim:
        return {"retrieval_score": 0.0, "relevant_evidence_count": 0}

    model = get_model()
    claim_embedding = model.encode([claim])
    evidence_embeddings = model.encode(evidence_texts)

    similarities = cosine_similarity(claim_embedding, evidence_embeddings)[0]
    max_similarity = np.max(similarities)
    relevant_count = np.sum(similarities >= threshold)

    return {
        "retrieval_score": float(max_similarity),
        "relevant_evidence_count": int(relevant_count)
    }
