"""
Source Trust Scoring for Evidence Pipeline — Part-9 refactor.

Uses URL-based domain extraction and exact-match against a tiered
publisher registry instead of the previous exploitable substring search.

Trust score mapping:
  Fact-check source  → 1.0
  Tier 1 publisher   → 1.0
  Tier 2 publisher   → 0.85
  Tier 3 (blogs)     → 0.50
  Wikipedia          → 0.65
  Unknown domain     → 0.40
"""

from app.utils.domain_utils import extract_and_normalize_domain
from .source_registry import TIER1_SOURCES, TIER2_SOURCES, TIER3_SOURCES, FACTCHECK_DOMAINS


def get_source_trust(url_or_name: str, source_type: str) -> float:
    """
    Returns a trust score (0.0–1.0) for an evidence source.

    Parameters
    ----------
    url_or_name : str
        The evidence URL (preferred) or publisher display name.
    source_type : str
        One of 'fact_check', 'knowledge', 'news', 'web', 'rss', etc.

    Security note
    -------------
    Domain lookup is exact-match only — no substring matching.
    A URL of 'http://fakereutersnews.com' produces domain
    'fakereutersnews.com', which is NOT in any trusted tier.
    """
    # Fact-check sources always get maximum trust regardless of domain
    if source_type == "fact_check":
        return 1.0

    # Attempt domain extraction from the URL
    domain = extract_and_normalize_domain(url_or_name)

    if domain and domain != "unknown":
        if domain in FACTCHECK_DOMAINS:
            return 1.0
        if domain in TIER1_SOURCES:
            return 1.0
        if domain in TIER2_SOURCES:
            return 0.85
        if domain in TIER3_SOURCES:
            return 0.50

    # Fallback: known source_type defaults
    type_defaults = {
        "knowledge": 0.65,    # Wikipedia-type sources
        "news":      0.55,    # Unknown news outlet
        "rss":       0.50,
        "web":       0.45,
        "factcheck_scrape": 0.70,
    }
    return type_defaults.get(source_type, 0.40)
