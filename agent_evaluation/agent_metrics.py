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

def claim_structure_match(predicted, ground_truth):
    """
    similarity(predicted_claim_structure, ground_truth_claim_structure)
    """
    model = get_model()
    # Flatten dict to string for embedding comparison if needed, 
    # or compare specific keys.
    p_str = str(sorted(predicted.items()))
    g_str = str(sorted(ground_truth.items()))
    
    embeddings = model.encode([p_str, g_str])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)

def entity_extraction_accuracy(predicted_entities, gt_entities):
    if not gt_entities:
        return 1.0 if not predicted_entities else 0.0
    
    p_set = set(e.lower() for e in predicted_entities)
    g_set = set(e.lower() for e in gt_entities)
    
    intersection = p_set.intersection(g_set)
    return len(intersection) / len(g_set)

def relation_classification_accuracy(predicted_relations, gt_relations):
    """
    predicted_relations: list of (evidence_id, label)
    gt_relations: list of (evidence_id, label)
    """
    if not gt_relations:
        return 1.0
    
    correct = 0
    gt_dict = dict(gt_relations)
    for ev_id, label in predicted_relations:
        if ev_id in gt_dict and label == gt_dict[ev_id]:
            correct += 1
            
    return correct / len(gt_relations)

def source_classification_accuracy(predicted_tiers, gt_tiers):
    """
    predicted_tiers: dict {domain: tier}
    gt_tiers: dict {domain: tier}
    """
    if not gt_tiers:
        return 1.0
    
    correct = 0
    for domain, tier in predicted_tiers.items():
        if domain in gt_tiers and tier == gt_tiers[domain]:
            correct += 1
            
    return correct / len(gt_tiers)

def temporal_consistency_accuracy(is_outdated_predicted, is_outdated_gt):
    return 1.0 if is_outdated_predicted == is_outdated_gt else 0.0

def evaluate_consensus_stability(verdict1, verdict2):
    """
    Run verification twice with shuffled evidence order and compare verdicts.
    """
    return 1.0 if verdict1 == verdict2 else 0.0
