"""
TrustLens Evidence-Based Credibility Engine

SCORING PHILOSOPHY (v2 — evidence-driven):
  The final credibility score is driven entirely by verifiable evidence signals.
  Heuristic proxies (source reputation, domain trust) have been removed from the
  scoring formula. Domain trust is still used downstream for evidence FILTERING
  (evidence_ranker.py) but must NOT influence credibilityScore.

NEW WEIGHTED FORMULA (v3):
  EvidenceSupportScore   × 0.50  — primary: LLM verification + trusted sources
  ClaimClarityScore      × 0.20  — claim structure / verifiability
  MediaAuthenticityScore × 0.20  — AI detection / metadata checks
  SemanticRiskScore      × 0.10  — manipulation signals / propaganda patterns

Weights sum to 1.00.  Final score is clamped to [0, 95].

ASYNC ARCHITECTURE (Phase 6 — unchanged):
  Phase 1: LLM text analysis (must complete first — primaryClaim needed)
  Phase 2+3+4: fact-check, domain (filtering only), image — CONCURRENT
  Phase 5: synchronous scoring + confidence (pure CPU)

  return_exceptions=True means one failing service does not abort others.
"""

import asyncio
import logging
from typing import Any

from app.services.domain_reputation_service import evaluate_domain
from app.services.image_authenticity_service import evaluate_image
from app.services.confidence_service import calculate_confidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redesigned Weights (v3)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "evidenceSupportScore":      0.50,
    "claimClarityScore":         0.20,
    "mediaAuthenticityScore":    0.20,
    "semanticRiskScore":         0.10,
}

# Neutral baselines
NEUTRAL_SCORES = {
    "evidenceSupportScore":      50.0,
    "claimClarityScore":         70.0,
    "mediaAuthenticityScore":    75.0,
    "semanticRiskScore":         100.0,
}

# Neutral constant for missing signals
DEFAULT_NEUTRAL = 65.0


# ---------------------------------------------------------------------------
# Neutral fallbacks — used when an async phase fails during gather
# ---------------------------------------------------------------------------

def _neutral_fact_check() -> dict[str, Any]:
    return {
        "factCheckScore": 65, "matchFound": False, "verdict": "No Match",
        "source": "", "referenceURL": "", "confidenceAdjustment": -0.1,
    }


def _neutral_domain() -> dict[str, Any]:
    """Domain still run for evidence filtering; never feeds credibilityScore."""
    return {
        "domainTrustScore": 65, "domain": None, "domainAgeDays": None,
        "httpsSecure": False, "isTrustedSource": False, "isBlacklisted": False,
        "riskFactors": ["Domain check unavailable"],
    }


def _neutral_image() -> dict[str, Any]:
    return {
        "imageAuthenticityScore": 75, "hashMatched": False, "matchedContext": None,
        "matchedEventDate": None, "contextMismatch": False, "aiGeneratedLikelihood": 0.0,
        "metadataPresent": False, "cameraMake": None, "cameraModel": None,
        "editingSoftwareDetected": False, "riskFactors": [],
    }


# ---------------------------------------------------------------------------
# Synchronous scoring (Phase 5 — CPU only, no I/O)
# ---------------------------------------------------------------------------

def calculate_credibility_score(scores: dict) -> float:
    """
    Pure function: compute the redesigned credibility score.
    """
    ev_support = max(0.0, min(100.0, float(scores.get("evidenceSupportScore", NEUTRAL_SCORES["evidenceSupportScore"]))))
    clarity    = max(0.0, min(100.0, float(scores.get("claimClarityScore",    NEUTRAL_SCORES["claimClarityScore"]))))
    media_auth = max(0.0, min(100.0, float(scores.get("mediaAuthenticityScore", NEUTRAL_SCORES["mediaAuthenticityScore"]))))
    risk       = max(0.0, min(100.0, float(scores.get("semanticRiskScore",     NEUTRAL_SCORES["semanticRiskScore"]))))

    final = (
        ev_support * WEIGHTS["evidenceSupportScore"]
        + clarity  * WEIGHTS["claimClarityScore"]
        + media_auth * WEIGHTS["mediaAuthenticityScore"]
        + risk     * WEIGHTS["semanticRiskScore"]
    )

    logger.info(
        "Score composition (v3) — evidence=%.1f clarity=%.1f media=%.1f risk=%.1f → final=%.2f",
        ev_support, clarity, media_auth, risk, final,
    )

    return float(round(float(max(0.0, min(95.0, final))), 2))


def compute_weighted_final_result(
    *,
    semantic_score: float | None = None,
    media_authenticity_score: float | None = None,
    domain_result: dict[str, Any] | None = None,
    image_auth_result: dict[str, Any] | None = None,
    manipulation_indicators: list | None = None,
    ai_video_probability: float | None = None,
    context_reuse_detected: bool = False,
    evidence_verification_score: float | None = None,
    fact_check_match: float | None = None,
    video_evidence_score: float | None = None,
    breaking_news_detected: bool = False,
    breaking_news_confidence: float | None = None,
    fact_check_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Synchronous: compute final credibility score and confidence breakdown.

    Why synchronous: pure arithmetic — no I/O whatsoever.
    Called after all async phases have resolved.

    NOTE: sourceReputationScore and domainTrustScore are NO LONGER PARAMETERS.
    Domain metadata (domain_result) is still accepted so it can be passed
    through to evidence_flags (for confidence modelling) and domainReputation
    in the API response, but it does not influence credibilityScore.
    """

    def _safe(val: int | float | None, key: str) -> float:
        if val is None:
            return float(NEUTRAL_SCORES.get(key, 60))
        try:
            return max(0.0, min(100.0, float(val)))
        except (TypeError, ValueError):
            return float(NEUTRAL_SCORES.get(key, 60))

    fc = fact_check_details or {}
    
    # Determine which visual signal to use (video has priority if present)
    visual_signal = video_evidence_score if video_evidence_score is not None else media_authenticity_score

    # Map inputs to new component keys
    component_scores = {
        "evidenceSupportScore":   _safe(evidence_verification_score, "evidenceSupportScore"),
        "claimClarityScore":      _safe(semantic_score, "claimClarityScore"),
        "mediaAuthenticityScore": _safe(visual_signal, "mediaAuthenticityScore"),
        "semanticRiskScore":      _safe(100.0 - (len(manipulation_indicators or []) * 20.0), "semanticRiskScore")
    }

    if breaking_news_detected and breaking_news_confidence is not None:
        component_scores["breakingNewsConfidence"] = _safe(breaking_news_confidence, "evidenceSupportScore")
        
        # Adjust weights: Fact checks are irrelevant for breaking news
        # Move verification weights into breakingNewsConfidence
        bn_weight = WEIGHTS["evidenceSupportScore"]
        
        bn_conf       = component_scores["breakingNewsConfidence"]
        clarity       = component_scores["claimClarityScore"]
        media_auth    = component_scores["mediaAuthenticityScore"]
        risk          = component_scores["semanticRiskScore"]
        
        final = (
            bn_conf    * bn_weight
            + clarity  * WEIGHTS["claimClarityScore"]
            + media_auth * WEIGHTS["mediaAuthenticityScore"]
            + risk     * WEIGHTS["semanticRiskScore"]
        )
        base_score = int(round(max(0.0, min(95.0, final))))
    else:
        base_score = int(round(calculate_credibility_score(component_scores)))

    # -----------------------------------------------------------------------
    # Penalty Layer — hard negative evidence (unchanged)
    # -----------------------------------------------------------------------
    _dr = domain_result or {}
    _ir = image_auth_result or {}

    if ai_video_probability is not None and ai_video_probability > 0.7:
        logger.warning("Phase B Penalty applied: AI video probability = %.2f", ai_video_probability)
        base_score = max(0, base_score - 40)

    if context_reuse_detected:
        logger.warning("Phase C Penalty applied: reused historical video context")
        base_score = max(0, base_score - 25)

    # -----------------------------------------------------------------------
    # Positive Boost Layer — reward verified evidence (domain removed)
    # -----------------------------------------------------------------------
    boost_applied = 0
    boost_reasons: list[str] = []

    eligible_for_boost = all(s >= 30 for s in component_scores.values())

    if eligible_for_boost:
        # 1. Verified evidence TRUE
        if component_scores.get("evidenceSupportScore", 0) >= 85:
            boost_applied += 10
            boost_reasons.append("Evidence Verification confirms claim")
        # 2. Visual evidence confirmed
        if component_scores.get("mediaAuthenticityScore", 0) >= 85:
            boost_applied += 5
            boost_reasons.append("Visual evidence confirmed")

    boost_applied = min(15, boost_applied)
    final_score = int(min(95, max(0, base_score + boost_applied)))

    # -----------------------------------------------------------------------
    # Phase 5: principled confidence (synchronous CPU-only)
    # Domain is kept in evidence_flags so confidence still benefits from it.
    # -----------------------------------------------------------------------
    evidence_flags = {
        "factCheckMatch":  bool(fc.get("matchFound")),
        "contextMismatch": bool(_ir.get("contextMismatch")),
        "imageReuseFound": bool(_ir.get("hashMatched")),
        "trustedDomain":   bool(_dr.get("isTrustedSource")),
    }
    confidence_result = calculate_confidence(component_scores, evidence_flags)

    def classify_score(score: int) -> tuple[str, str]:
        if score >= 85:   return "Minimal",    "Highly Reliable"
        elif score >= 70: return "Low",         "Reliable"
        elif score >= 50: return "Low-Medium",  "Likely Reliable"
        elif score >= 30: return "Medium",      "Questionable"
        else:             return "High",        "High Risk"

    risk_level, final_verdict = classify_score(final_score)

    logger.info(
        "Final credibility — score=%d verdict=%s riskLevel=%s boost=%d boostReasons=%s",
        final_score, final_verdict, risk_level, boost_applied, boost_reasons,
    )

    result_dict = {
        "componentScores":      component_scores,
        "factCheckDetails":     fc,
        "baseWeightedScore":    base_score,
        "positiveBoostApplied": boost_applied,
        "boostReasons":         boost_reasons,
        "finalScore":           final_score,
        "finalVerdict":         final_verdict,
        "riskLevel":            risk_level,
        "confidence":           confidence_result["confidenceScore"],
        "confidenceLevel":      confidence_result["confidenceLevel"],
        "confidenceBreakdown":  confidence_result,
    }
    
    if breaking_news_detected:
        result_dict["breakingNewsDetected"] = True
        
    return result_dict


# ---------------------------------------------------------------------------
# Async orchestrator — Phase 6 parallel execution
# ---------------------------------------------------------------------------

async def compute_full_credibility(
    text_analysis: dict[str, Any] | None,
    image_analysis: dict[str, Any] | None,
    video_analysis: dict[str, Any] | None = None,
    evidence_results: list[dict[str, Any]] | None = None,
    source_url: str | None = None,
    image_bytes: bytes | None = None,
) -> dict[str, Any]:
    """
    Async: Run the full Phase 6 credibility pipeline.

    Execution order:
      Phase 1: LLM text analysis (awaited in ROUTE before this is called)
      Phase 2+3+4: fact-check, domain (filtering only), image — CONCURRENT
      Phase 5: synchronous scoring + confidence

    Domain Phase (3) still runs for evidence filtering downstream but its
    score is NOT included in the credibility formula.
    """
    primary_claim = ""
    semantic_score: float | None = None
    manipulation_indicators: list | None = None
    knowledge_support_score: float | None = None
    breaking_news_detected = False
    breaking_news_confidence: float | None = None

    if text_analysis and text_analysis.get("status") != "skipped":
        semantic = text_analysis.get("semantic") or {}
        primary_claim = (semantic.get("primaryClaim") or "").strip()
        semantic_score = text_analysis.get("credibilityScore") or semantic.get("semanticScore")
        manipulation_indicators = semantic.get("manipulationIndicators", [])
        
        kv = text_analysis.get("knowledgeVerification", {})
        knowledge_support_score = kv.get("knowledgeSupportScore")
        
        breaking_news_detected = kv.get("breakingNewsDetected", False)
        # Handle breaking news confidence later after evidence_verification_score is calculated
    # ---- Extract video evidence score (new: dedicated signal, not blended into semantic) ----
    video_evidence_score: float | None = None
    ai_video_probability: float | None = None
    context_reuse_detected = False

    if video_analysis and video_analysis.get("status") != "skipped":
        raw_video_score = video_analysis.get("credibilityScore")
        if raw_video_score is not None:
            # Video score feeds videoEvidenceScore directly (25 % weight)
            video_evidence_score = float(raw_video_score)
            # Apply an internal pre-penalty for very-low scores to prevent
            # video result from being silently neutral.
            if video_evidence_score < 40:
                video_evidence_score = max(0.0, video_evidence_score - 10)

        ai_detection = video_analysis.get("aiDetection", {})
        ai_video_probability = ai_detection.get("aiGeneratedProbability")
        context_detection = video_analysis.get("contextDetection", {})
        context_reuse_detected = context_detection.get("contextReuseDetected", False)

        # Video-based knowledge support overrides text-based if present
        v_knowledge = video_analysis.get("knowledgeVerification", {}).get("knowledgeSupportScore")
        if v_knowledge is not None:
            knowledge_support_score = v_knowledge

    # ---- Extract image signals ----
    image_auth_score_override: float | None = None
    if image_analysis and image_analysis.get("status") != "skipped":
        img_score = image_analysis.get("credibilityScore")
        if img_score is not None:
            image_auth_score_override = float(img_score)
            # Blend semantic with image when image is the primary content
            if semantic_score is not None:
                semantic_score = (semantic_score * 0.2) + (image_auth_score_override * 0.8)
                if image_auth_score_override < 40:
                    semantic_score -= 10
            else:
                semantic_score = image_auth_score_override
                if image_auth_score_override < 40:
                    semantic_score -= 10

    # ---- Phase 2 + 3 + 4: concurrent I/O ----
    domain_result, image_auth_result = await asyncio.gather(
        evaluate_domain(source_url),
        asyncio.to_thread(evaluate_image, image_bytes, primary_claim or None),
        return_exceptions=True,
    )
    fc_raw = {"claims": []}

    if isinstance(fc_raw, Exception):
        logger.warning("Fact-check phase failed: %s", fc_raw)
        fc_raw = {"claims": []}
    if isinstance(domain_result, Exception):
        logger.warning("Domain phase failed: %s", domain_result)
        domain_result = _neutral_domain()
    if isinstance(image_auth_result, Exception):
        logger.warning("Image phase failed: %s", image_auth_result)
        image_auth_result = _neutral_image()

    # ---- Phase D: Process Unified Evidence Results ----
    evidence_verification_score = 50.0
    evidence_sources_used = []
    verified_claims = []
    verification_breakdown = []

    if evidence_results:
        scores = []
        for res in evidence_results:
            v_data = res.get("verification", {})
            v = v_data.get("verdict", "UNVERIFIED")
            
            verified_claims.append({
                "claim": res.get("claim", ""),
                "verdict": v
            })
            
            verification_breakdown.append({
                "claim": res.get("claim", ""),
                "verdict": v,
                "confidence": v_data.get("confidence", 0.0),
                "reasoning": v_data.get("reasoning", "")
            })

            if v == "SUPPORTED":
                scores.append(100.0)
            elif v == "CONTRADICTED":
                scores.append(0.0)
            else:
                scores.append(50.0)
            
            for doc in res.get("evidence", []):
                source_name = doc.get("source", "Unknown")
                if source_name not in evidence_sources_used:
                    evidence_sources_used.append(source_name)
        
        if scores:
            evidence_verification_score = sum(scores) / len(scores)

    if breaking_news_detected:
        breaking_news_confidence = evidence_verification_score
        logger.info("Breaking news detected. Verification score %.1f used as confidence.", breaking_news_confidence)

    # ---- Phase 5: synchronous scoring + confidence ----
    dr = domain_result if isinstance(domain_result, dict) else _neutral_domain()
    ia = image_auth_result if isinstance(image_auth_result, dict) else _neutral_image()

    media_authenticity_score = image_auth_score_override if image_auth_score_override is not None else video_evidence_score

    weighted_result = compute_weighted_final_result(
        semantic_score=semantic_score,
        media_authenticity_score=media_authenticity_score,
        domain_result=dr,
        image_auth_result=ia,
        manipulation_indicators=manipulation_indicators,
        ai_video_probability=ai_video_probability,
        context_reuse_detected=context_reuse_detected,
        evidence_verification_score=evidence_verification_score,
        fact_check_match=0.0, # Now unused in v3 formula
        video_evidence_score=video_evidence_score,
        breaking_news_detected=breaking_news_detected,
        breaking_news_confidence=breaking_news_confidence,
        fact_check_details={"matchFound": False},
    )

    # Explainability Data
    weighted_result["evidenceSourcesUsed"] = evidence_sources_used
    weighted_result["verifiedClaims"] = verified_claims
    weighted_result["verificationBreakdown"] = verification_breakdown

    weighted_result["domainReputation"]  = domain_result
    weighted_result["imageAuthenticity"] = image_auth_result
    return weighted_result


async def _coro_empty_fact() -> dict:
    """Async no-op yielding an empty fact-check response (no API call made)."""
    return {"claims": [], "nextPageToken": ""}
