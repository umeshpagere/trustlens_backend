def is_valid_claim(claim: str) -> bool:
    """
    Validate claim structure (subject + action).
    """

    if not claim:
        return False

    words = claim.split()

    if len(words) < 3:
        return False

    verbs = [
        "said", "announced", "launched", "joined",
        "confirmed", "reported", "declared", "banned",
        "arrested", "won", "caused", "triggered",
        "is", "are", "was", "were", "happened", "occurred",
        "showed", "killed", "passed", "signed", "stated",
        "claimed", "made", "did", "does", "do", "has", "have", "had"
    ]
    
    # Using a slightly expanded verb list to avoid dropping valid claims too aggressively
    return any(v in claim.lower() for v in verbs)
