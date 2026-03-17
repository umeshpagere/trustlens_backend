def evaluate_credibility(score, expected_range):
    """
    Evaluates if the credibility score falls within the expected range.
    """
    if score is None or not expected_range:
        return {"credibility_score_accuracy": 0.0}
        
    min_s, max_s = expected_range
    is_correct = min_s <= score <= max_s
    
    return {
        "credibility_score_accuracy": 1.0 if is_correct else 0.0
    }
