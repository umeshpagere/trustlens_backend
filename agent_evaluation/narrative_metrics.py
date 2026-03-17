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

def cluster_coherence(cluster_claims):
    """
    mean_cosine_similarity(cluster_claims)
    """
    if len(cluster_claims) < 2:
        return 1.0
    
    model = get_model()
    embeddings = model.encode(cluster_claims)
    
    sim_matrix = cosine_similarity(embeddings)
    # Extract upper triangle excluding diagonal
    upper_tri = sim_matrix[np.triu_indices(len(cluster_claims), k=1)]
    return float(np.mean(upper_tri))

def campaign_detection_accuracy(detected_campaigns, true_campaigns):
    """
    precision = detected_clusters ∩ true_clusters / detected_clusters
    recall = detected_clusters ∩ true_clusters / true_clusters
    """
    d_set = set(detected_campaigns)
    t_set = set(true_campaigns)
    
    if not d_set and not t_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    intersection = d_set.intersection(t_set)
    precision = len(intersection) / len(d_set) if d_set else 0.0
    recall = len(intersection) / len(t_set) if t_set else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
