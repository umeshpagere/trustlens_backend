from agent_evaluation.utils.similarity_utils import compute_semantic_similarity

def evaluate_query_generation(queries, claim):
    """
    Evaluates the relevance of generated queries to the claim.
    """
    if not queries or not claim:
        return {"query_relevance_score": 0.0}
    
    scores = []
    for q in queries:
        scores.append(compute_semantic_similarity(q, claim))
        
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "query_relevance_score": avg_score
    }
