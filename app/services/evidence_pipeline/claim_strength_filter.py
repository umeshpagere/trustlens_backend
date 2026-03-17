import re
import logging

logger = logging.getLogger(__name__)

def is_metadata_claim(claim: str) -> bool:
    """Detect claims that are likely platform metadata or instructions."""
    metadata_patterns = [
        "posted on",
        "uploaded on",
        "this video",
        "this post",
        "watch this",
        "caption reads",
        "posted at",
        "tweeted on",
        "shared on",
        "clip shows",
        "image shows",
        "screenshot of"
    ]
    
    claim_lower = claim.lower()
    for pattern in metadata_patterns:
        if pattern in claim_lower:
            return True
            
    return False

def is_question(claim: str) -> bool:
    """Identify claims that are interrogative rather than declarative."""
    return claim.strip().endswith("?")

def is_short_claim(claim: str) -> bool:
    """Identify claims that likely lack sufficient context for verification."""
    return len(claim.split()) < 3

def filter_claims(claims: list[str]) -> list[str]:
    """
    Filter a list of claims to remove weak, vague, or metadata-heavy entries.
    Returns only strong factual claims suitable for evidence retrieval.
    """
    if not claims:
        return []

    filtered = []
    rejected = []

    for claim in claims:
        claim_stripped = claim.strip()
        
        # 1. Reject questions
        if is_question(claim_stripped):
            rejected.append((claim, "Question"))
            continue
            
        # 2. Reject metadata or platform context
        if is_metadata_claim(claim_stripped):
            rejected.append((claim, "Metadata/Platform Context"))
            continue
            
        # 3. Reject short claims (lack subject+verb+object usually)
        if is_short_claim(claim_stripped):
            rejected.append((claim, "Too short/vague"))
            continue

        filtered.append(claim_stripped)

    # Log results for debugging
    if rejected:
        logger.info(f"Claim strength filter rejected {len(rejected)} claims:")
        for c, reason in rejected:
            logger.info(f"  - [{reason}] {c}")
    
    logger.info(f"Claims remaining after filtering: {len(filtered)}")
    
    return filtered
