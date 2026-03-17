def evaluate_verdict(predicted_verdict, ground_truth_verdict):
    """
    Evaluates verdict accuracy with basic normalization.
    """
    VERDICT_MAP = {
        "TRUE": "SUPPORTED",
        "FALSE": "CONTRADICTED",
        "NOT_ENOUGH_INFO": "UNVERIFIED"
    }
    
    if not predicted_verdict or not ground_truth_verdict:
        return {"verdict_accuracy": 0.0}

    p = VERDICT_MAP.get(str(predicted_verdict).upper(), str(predicted_verdict).upper())
    g = VERDICT_MAP.get(str(ground_truth_verdict).upper(), str(ground_truth_verdict).upper())
    
    return {
        "verdict_accuracy": 1.0 if p == g else 0.0
    }
