"""
Coverage Metrics — Part-12

Computes a single 0.0–1.0 coverage score indicating how much useful
evidence has been retrieved for a claim.  Used by the adaptive
retrieval controller to decide whether a second retrieval loop is needed.

Formula:
    coverage = 0.40 * domain_diversity
             + 0.30 * evidence_volume
             + 0.20 * support_contradiction_balance
             + 0.10 * source_trust_average
"""

import logging
from app.services.evidence_pipeline.source_registry import (
    TIER1_SOURCES, TIER2_SOURCES, FACTCHECK_DOMAINS
)

logger = logging.getLogger(__name__)

# --- Tunable thresholds -------------------------------------------------------
COVERAGE_SUFFICIENT  = 0.40   # above this → stop retrieval
MAX_VOLUME_DOCS      = 30     # beyond this, extra docs add diminishing value
MIN_DOMAIN_DIVERSITY = 5      # target number of distinct domains


def compute_coverage(evidence_docs: list) -> float:
    """
    Compute a coverage score (0.0–1.0) for a list of evidence docs.

    Args:
        evidence_docs: raw list of evidence dicts from aggregate_evidence()

    Returns:
        float in [0.0, 1.0]
    """
    if not evidence_docs:
        return 0.0

    total = len(evidence_docs)

    # 1. Domain diversity — how many unique trusted domains are represented
    domains = {doc.get("domain", "unknown") for doc in evidence_docs if doc.get("domain")}
    domain_diversity = min(len(domains) / MIN_DOMAIN_DIVERSITY, 1.0)

    # 2. Evidence volume — normalised against target document count
    evidence_volume = min(total / MAX_VOLUME_DOCS, 1.0)

    # 3. Support / contradiction balance
    #    We use NLI labels when available; otherwise fall back to type heuristics
    support_count    = sum(
        1 for d in evidence_docs
        if d.get("nli_label") == "SUPPORTED"
        or d.get("type") == "fact_check"
    )
    contradict_count = sum(
        1 for d in evidence_docs
        if d.get("nli_label") == "CONTRADICTED"
    )
    has_support    = support_count > 0
    has_contradict = contradict_count > 0

    if has_support and has_contradict:
        balance = 1.0        # ideal: both sides represented
    elif has_support or has_contradict:
        balance = 0.60       # only one side
    else:
        balance = 0.30       # nothing useful yet

    # 4. Source trust average — tier-based
    trust_scores = []
    for doc in evidence_docs:
        domain = doc.get("domain", "")
        ts     = doc.get("trust_score")
        if ts is not None:
            trust_scores.append(float(ts))
        elif domain in TIER1_SOURCES or domain in FACTCHECK_DOMAINS:
            trust_scores.append(1.0)
        elif domain in TIER2_SOURCES:
            trust_scores.append(0.75)
        else:
            trust_scores.append(0.40)

    trust_average = sum(trust_scores) / len(trust_scores) if trust_scores else 0.4

    coverage = (
        0.40 * domain_diversity      +
        0.30 * evidence_volume       +
        0.20 * balance               +
        0.10 * trust_average
    )
    coverage = round(max(0.0, min(1.0, coverage)), 3)

    logger.debug(
        f"[Coverage] total={total} domains={len(domains)} "
        f"domain_div={domain_diversity:.2f} vol={evidence_volume:.2f} "
        f"balance={balance:.2f} trust={trust_average:.2f} → {coverage}"
    )
    return coverage


def is_coverage_sufficient(evidence_docs: list) -> tuple[bool, float]:
    """
    Returns (sufficient: bool, score: float).
    Sufficient if score >= threshold OR doc count >= 8 (volume shortcut).
    """
    score = compute_coverage(evidence_docs)
    sufficient = score >= COVERAGE_SUFFICIENT or len(evidence_docs) >= 8
    return sufficient, score
