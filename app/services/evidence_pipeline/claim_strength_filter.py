import re
import logging

logger = logging.getLogger(__name__)

# Patterns that indicate product listing details — not news-verifiable claims
UNVERIFIABLE_PATTERNS = [
    r'₹[\d,]+',                          # Indian rupee prices
    r'Rs\.?\s*[\d,]+',                   # Rs. prices
    r'\$[\d,]+',                          # USD prices
    r'\bsize\s+(is\s+)?\d',              # "size is 6" / "size 6"
    r'\bcolou?r\s+is\b',                  # "color is black"
    r'\bdelivery\s+in\s+\d+\s+day',      # "delivery in 2 days"
    r'\bdelivers?\s+by\b',               # "delivers by Thursday"
    r'\bcoins?\b',                        # loyalty coins/points
    r'\bdiscount\b',                      # discount claims
    r'\bassured\b',                       # platform badges
    r'\boriginal\s+price\b',             # pricing details
    r'\boption\s+to\s+pay\b',            # payment options
    r'\brating\s+of\s+[\d.]+\s+out\s+of\b',  # star ratings
    r'\b\d+(\.\d+)?\s+out\s+of\s+\d+\b', # "4.5 out of 5"
    r'\bfree\s+deliver',                  # "free delivery"
    r'\bseller\b',                        # seller info
    r'\bstock\b',                         # in stock / out of stock
    r'\badd\s+to\s+(cart|bag|wishlist)\b',# e-commerce CTAs
    r'\bproduct\s+is\s+(available|listed|marked|assu)',  # product status
]


def is_unverifiable_product_detail(claim: str) -> bool:
    """Return True if the claim is an e-commerce/product listing detail
    that cannot be verified by news sources or fact-checkers."""
    text_lower = claim.lower()
    for pattern in UNVERIFIABLE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    # Also reject claims with fewer than 5 meaningful words
    words = [w for w in claim.split() if len(w) > 2]
    if len(words) < 5:
        return True
    return False

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

        # 4. Reject product listing / e-commerce details
        if is_unverifiable_product_detail(claim_stripped):
            rejected.append((claim, "Product/e-commerce detail"))
            continue

        filtered.append(claim_stripped)

    # Log results for debugging
    if rejected:
        logger.info(f"Claim strength filter rejected {len(rejected)} claims:")
        for c, reason in rejected:
            logger.info(f"  - [{reason}] {c}")
    
    logger.info(f"Claims remaining after filtering: {len(filtered)}")
    
    return filtered
