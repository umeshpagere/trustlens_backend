"""
Source Trust Weights for Evidence Pipeline.
This module defines the trust scores for various evidence types and specific sources.
"""

SOURCE_TRUST = {
    "fact_check": 1.0,
    "reuters": 0.9,
    "bbc": 0.9,
    "associated_press": 0.9,
    "news": 0.7,
    "wikipedia": 0.6,
    "unknown": 0.5
}

def get_source_trust(source_name: str, source_type: str) -> float:
    """
    Returns the trust score for a given source name and type.
    """
    # 1. Check if source name is specifically listed (case-insensitive)
    name_lower = source_name.lower()
    for key, weight in SOURCE_TRUST.items():
        if key in name_lower and key not in ["fact_check", "news", "wikipedia", "unknown"]:
            return weight
            
    # 2. Check source type
    return SOURCE_TRUST.get(source_type, SOURCE_TRUST["unknown"])
