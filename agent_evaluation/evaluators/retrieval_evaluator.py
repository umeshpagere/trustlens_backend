def evaluate_retrieval(ranked_evidence, reference_evidence):
    """
    Evaluates retrieval performance using reference evidence.
    """
    if not reference_evidence:
        return {"retrieval_score": 1.0, "context_precision": 1.0, "context_recall": 1.0}
    
    if not ranked_evidence:
        return {"retrieval_score": 0.0, "context_precision": 0.0, "context_recall": 0.0}

    retrieved_texts = [e.get("text", "").lower() for e in ranked_evidence]
    reference_texts = [r.lower() for r in reference_evidence]
    
    relevant_count = 0
    for ref in reference_texts:
        # Check if reference text is found in any retrieved document
        if any(ref in ret or ret in ref for ret in retrieved_texts):
            relevant_count += 1
            
    precision = relevant_count / len(retrieved_texts) if retrieved_texts else 0.0
    recall = relevant_count / len(reference_texts) if reference_texts else 0.0
    
    # f1-style retrieval score
    score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "retrieval_score": score,
        "context_precision": precision,
        "context_recall": recall
    }
