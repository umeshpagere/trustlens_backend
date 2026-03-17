from datetime import datetime

def compute_temporal_gap(claim_time, evidence_time):
    if not claim_time or not evidence_time:
        return None
    
    # If claim_time or evidence_time are strings in ISO format, parse them.
    if isinstance(claim_time, str):
        try:
            claim_time = datetime.fromisoformat(claim_time.replace("Z", "+00:00"))
        except ValueError:
            pass
            
    if isinstance(evidence_time, str):
        try:
            # Handle wikipedia "2026-03-12T19:00:23Z" format
            evidence_time = datetime.fromisoformat(evidence_time.replace("Z", "+00:00"))
        except ValueError:
            pass

    if not isinstance(claim_time, datetime) or not isinstance(evidence_time, datetime):
        return None

    return abs((claim_time - evidence_time).days)

def classify_temporal_gap(gap_days: int) -> str:
    if gap_days is None:
        return "UNKNOWN"
    if gap_days <= 1:
        return "RECENT"
    elif gap_days <= 30:
        return "CURRENT"
    elif gap_days <= 365:
        return "OLD"
    else:
        return "HISTORICAL"
