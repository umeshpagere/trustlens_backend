from agent_evaluation.utils.similarity_utils import compute_semantic_similarity

def evaluate_claim_extraction(predicted_claims, ground_truth_claims):
    """
    Evaluates claim extraction quality.
    
    Args:
        predicted_claims (list[str]): Claims extracted by the agent.
        ground_truth_claims (list[str]): Expected claims from dataset.
        
    Returns:
        dict: {
            "claim_extraction_accuracy": bool,
            "claim_semantic_similarity": float
        }
    """
    if not predicted_claims or not ground_truth_claims:
        return {"claim_extraction_accuracy": False, "claim_semantic_similarity": 0.0}
    
    # Compare primary claims
    pred = predicted_claims[0]
    gt = ground_truth_claims[0]
    
    similarity = compute_semantic_similarity(pred, gt)
    
    return {
        "claim_extraction_accuracy": similarity > 0.85,
        "claim_semantic_similarity": similarity
    }
