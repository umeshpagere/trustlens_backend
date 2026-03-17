"""
Source Authority Registry for TrustLens evidence credibility scoring.

Publishers are grouped into tiers based on editorial standards,
journalistic credibility, and fact-checking track record.
Domain matching is exact (not substring) to prevent spoofing attacks.
"""

# ---------------------------------------------------------------------------
# Tier 1 — High authority: major wire services and established news orgs
# ---------------------------------------------------------------------------
TIER1_SOURCES: set[str] = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "nytimes.com",
    "theguardian.com",
    "bloomberg.com",
    "wsj.com",
    "washingtonpost.com",
    "economist.com",
    "ft.com",              # Financial Times
    "npr.org",
    "pbs.org",
    "who.int",
    "cdc.gov",
    "gov.uk",
    "europa.eu",
    "un.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
}

# ---------------------------------------------------------------------------
# Tier 2 — Standard authority: established outlets with editorial review
# ---------------------------------------------------------------------------
TIER2_SOURCES: set[str] = {
    "cnn.com",
    "nbcnews.com",
    "cbsnews.com",
    "abcnews.go.com",
    "aljazeera.com",
    "forbes.com",
    "time.com",
    "usatoday.com",
    "politico.com",
    "theatlantic.com",
    "vox.com",
    "axios.com",
    "businessinsider.com",
    "huffpost.com",
    "newsweek.com",
    "latimes.com",
    "chicagotribune.com",
    "telegraph.co.uk",
    "independent.co.uk",
    "thetimes.co.uk",
    "dw.com",              # Deutsche Welle
    "france24.com",
    "euronews.com",
    "thehindu.com",
    "dawn.com",
    "abc.net.au",
}

# ---------------------------------------------------------------------------
# Tier 3 — Low authority: blogs, personal sites, user-generated content
# ---------------------------------------------------------------------------
TIER3_SOURCES: set[str] = {
    "medium.com",
    "substack.com",
    "blogspot.com",
    "wordpress.com",
    "tumblr.com",
    "livejournal.com",
    "rumble.com",
    "bitchute.com",
    "brandnewtube.com",
    "naturalnews.com",
    "infowars.com",
    "beforeitsnews.com",
    "zerohedge.com",
}

# ---------------------------------------------------------------------------
# Fact-check authority registries — sites always treated as Fact-Check type
# ---------------------------------------------------------------------------
FACTCHECK_DOMAINS: set[str] = {
    "snopes.com",
    "factcheck.org",
    "politifact.com",
    "fullfact.org",
    "reuters.com",           # Reuters fact-check section
    "apnews.com",            # AP fact checks
    "boomlive.in",
    "altnews.in",
    "vishvasnews.com",
    "africacheck.org",
    "chequeado.com",
}


def get_source_tier(domain: str) -> int:
    """
    Return the trust tier for a domain: 1, 2, 3, or 0 (unknown).
    Uses exact domain matching — no substring matching.
    """
    if domain in TIER1_SOURCES or domain in FACTCHECK_DOMAINS:
        return 1
    if domain in TIER2_SOURCES:
        return 2
    if domain in TIER3_SOURCES:
        return 3
    return 0
