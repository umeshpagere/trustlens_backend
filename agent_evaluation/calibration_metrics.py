import numpy as np

def compute_calibration_error(predictions, actuals, bins=5):
    """
    Computes Expected Calibration Error (ECE) for credibility/confidence scores.
    predictions: list of scores (0-100)
    actuals: list of binary outcomes (1 for correct, 0 for incorrect)
    """
    if not predictions:
        return 0.0
    
    preds = np.array(predictions) / 100.0
    acts = np.array(actuals)
    
    bin_boundaries = np.linspace(0, 1, bins + 1)
    ece = 0.0
    
    for i in range(bins):
        bin_idx = (preds >= bin_boundaries[i]) & (preds < bin_boundaries[i+1])
        if np.any(bin_idx):
            bin_acc = np.mean(acts[bin_idx])
            bin_conf = np.mean(preds[bin_idx])
            bin_weight = np.sum(bin_idx) / len(preds)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def evaluate_calibration(samples):
    """
    samples: list of dicts with {'score': 0-100, 'correct': bool}
    """
    scores = [s['score'] for s in samples]
    outcomes = [1 if s['correct'] else 0 for s in samples]
    
    return compute_calibration_error(scores, outcomes)
