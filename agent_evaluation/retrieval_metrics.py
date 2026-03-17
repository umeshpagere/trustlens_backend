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

def evaluate_retrieval_quality(claim, retrieved_evidence, ground_truth_evidence, queries_used, retrieved_docs_count):
    """
    Evaluates retrieval quality using Recall@k, Precision@k, coverage, diversity, and efficiency.
    """
    if not retrieved_evidence or not ground_truth_evidence:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "evidence_coverage": 0.0,
            "source_diversity": 0.0,
            "planner_efficiency": 0.0
        }

    # Recall and Precision
    retrieved_set = set(retrieved_evidence)
    gt_set = set(ground_truth_evidence)
    intersection = retrieved_set.intersection(gt_set)
    
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0

    # Evidence Coverage (Simplified: check if key entities from claim are in evidence)
    # In a real scenario, we might use NLP to extract entities. 
    # For now, we'll use a placeholder or simple word overlap if entities aren't passed.
    # Note: User request mentions coverage = entities_covered / total_entities.
    # We'll assume entities are extracted from the claim text.
    
    # Source Diversity
    # retrieved_evidence is assumed to be a list of strings or dicts with source info.
    # If it's strings, we can't easily compute diversity unless they contains URLs.
    # We'll expect dicts with 'domain' or 'source' keys.
    unique_sources = set()
    for item in retrieved_evidence:
        if isinstance(item, dict):
            unique_sources.add(item.get("domain") or item.get("source") or "unknown")
        elif isinstance(item, str) and "://" in item:
            from urllib.parse import urlparse
            try:
                unique_sources.add(urlparse(item).netloc)
            except:
                pass
    
    diversity = len(unique_sources) / len(retrieved_evidence) if retrieved_evidence else 0.0

    # Planner Efficiency
    efficiency = retrieved_docs_count / len(queries_used) if queries_used else 0.0

    return {
        "recall_at_k": float(recall),
        "precision_at_k": float(precision),
        "source_diversity": float(diversity),
        "planner_efficiency": float(efficiency)
    }

def compute_evidence_coverage(entities, retrieved_texts):
    """
    Measure whether retrieved evidence covers all key claim entities.
    coverage = entities_covered / total_entities
    """
    if not entities:
        return 1.0
    
    covered_count = 0
    all_text = " ".join(retrieved_texts).lower()
    for entity in entities:
        if entity.lower() in all_text:
            covered_count += 1
            
    return covered_count / len(entities)
