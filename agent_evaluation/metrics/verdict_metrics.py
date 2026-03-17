VERDICT_MAP = {
    "TRUE": "SUPPORTED",
    "FALSE": "CONTRADICTED",
    "MISLEADING": "CONTRADICTED",
    "NOT_ENOUGH_INFO": "UNVERIFIED",
    "REFUTED": "CONTRADICTED",
    "SUPPORTED": "SUPPORTED",
    "CONTRADICTED": "CONTRADICTED",
    "UNVERIFIED": "UNVERIFIED"
}

def evaluate_verdict(predicted_verdict, ground_truth_verdict):
    """
    Compares predicted verdict with ground truth verdict using tolerance mapping.
    """
    if not predicted_verdict or not ground_truth_verdict:
        return {"verdict_correct": False}

    p = VERDICT_MAP.get(str(predicted_verdict).strip().upper(), str(predicted_verdict).strip().upper())
    g = VERDICT_MAP.get(str(ground_truth_verdict).strip().upper(), str(ground_truth_verdict).strip().upper())

    return {
        "verdict_correct": p == g
    }

def compute_evidence_grounding_score(explanation_sentences, evidence_texts):
    """
    Measure whether explanations reference real evidence.
    grounded_sentences / total_explanation_sentences
    """
    if not explanation_sentences:
        return 0.0
    
    grounded_count = 0
    all_evidence = " ".join(evidence_texts).lower()
    
    for sentence in explanation_sentences:
        # Simplified grounding check: see if major keywords or whole sentence concepts
        # (excluding common words) are present in evidence.
        # For a robust version, we'd use embedding similarity.
        if any(word in all_evidence for word in sentence.lower().split() if len(word) > 4):
            grounded_count += 1
            
    return grounded_count / len(explanation_sentences)

def compute_unsupported_reasoning_rate(reasoning_steps, evidence_texts):
    """
    Detect reasoning that references evidence not present in context.
    unsupported_statements / total_reasoning_steps
    """
    if not reasoning_steps:
        return 0.0
    
    unsupported_count = 0
    all_evidence = " ".join(evidence_texts).lower()
    
    for step in reasoning_steps:
        # If the reasoning step mentions a fact not in evidence
        # This is hard to detect without NLI or LLM.
        # For now, we'll use a placeholder logic or simple overlap.
        # Ideally, we'd use an NLI model here.
        pass
        
    return unsupported_count / len(reasoning_steps)
