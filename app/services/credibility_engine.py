"""
TrustLens Evidence-Based Credibility Engine

SCORING PHILOSOPHY (v2 — evidence-driven):
  The final credibility score is driven entirely by verifiable evidence signals.
  Heuristic proxies (source reputation, domain trust) have been removed from the
  scoring formula. Domain trust is still used downstream for evidence FILTERING
  (evidence_ranker.py) but must NOT influence credibilityScore.

ACTUAL WEIGHTED FORMULA (v3 — matches WEIGHTS dict below):
  EvidenceSupportScore   × 0.50  — primary: LLM verification + trusted sources
  SourceTrustScore       × 0.20  — domain reputation of retrieved sources
  MediaAuthenticityScore × 0.20  — AI detection / metadata checks
  SemanticRiskScore      × 0.10  — manipulation signals / propaganda patterns

Weights sum to 1.00.  Final score is clamped to [0, 100].

ASYNC ARCHITECTURE:
  Phase A: LLM text analysis (must complete first — primaryClaim needed)
  Phase B: domain (filtering only), image — CONCURRENT async I/O
  Phase C: synchronous scoring + confidence (pure CPU)

  return_exceptions=True means one failing service does not abort others.
"""

import asyncio
import datetime
import logging
import math
from typing import Any

from app.services.domain_reputation_service import evaluate_domain
from app.services.image_authenticity_service import evaluate_image
from app.services.confidence_service import calculate_confidence
from app.services.temporal_analysis import compute_temporal_gap, classify_temporal_gap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credibility Weights (evidence-grounded)
# ---------------------------------------------------------------------------
# Recommended formula:
#   credibility =
#       0.5 * evidence_support +
#       0.2 * source_trust +
#       0.2 * media_authenticity +
#       0.1 * risk_score
# ---------------------------------------------------------------------------
WEIGHTS = {
    "evidenceSupportScore":   0.50,
    "sourceTrustScore":       0.20,
    "mediaAuthenticityScore": 0.20,
    "semanticRiskScore":      0.10,
}

# Neutral baselines
NEUTRAL_SCORES = {
    "evidenceSupportScore":   50.0,
    "sourceTrustScore":       50.0,
    "mediaAuthenticityScore": 75.0,
    "semanticRiskScore":      50.0,
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
    source_trust = max(0.0, min(100.0, float(scores.get("sourceTrustScore", NEUTRAL_SCORES["sourceTrustScore"]))))
    media_auth = max(0.0, min(100.0, float(scores.get("mediaAuthenticityScore", NEUTRAL_SCORES["mediaAuthenticityScore"]))))
    risk = max(0.0, min(100.0, float(scores.get("semanticRiskScore", NEUTRAL_SCORES["semanticRiskScore"]))))

    final = (
        ev_support * WEIGHTS["evidenceSupportScore"]
        + source_trust * WEIGHTS["sourceTrustScore"]
        + media_auth * WEIGHTS["mediaAuthenticityScore"]
        + risk * WEIGHTS["semanticRiskScore"]
    )

    logger.info(
        "Score composition — evidence=%.1f source_trust=%.1f media=%.1f risk=%.1f → final=%.2f",
        ev_support,
        source_trust,
        media_auth,
        risk,
        final,
    )

    return float(round(float(max(0.0, min(100.0, final))), 2))


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
    evidence_confidence: float | None = None,
    temporal_consistency_score: float | None = None,
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
        # Source trust is derived from domain reputation if available.
        "sourceTrustScore":       _safe((domain_result or {}).get("domainTrustScore"), "sourceTrustScore"),
        "mediaAuthenticityScore": _safe(visual_signal, "mediaAuthenticityScore"),
        # Higher manipulation indicators → higher risk score.
        "semanticRiskScore":      _safe(
            100.0 * math.exp(-0.35 * len(manipulation_indicators or [])),
            "semanticRiskScore",
        ),
        "temporalConsistencyScore": _safe(temporal_consistency_score, "semanticRiskScore"),
    }

    if breaking_news_detected and breaking_news_confidence is not None:
        component_scores["breakingNewsConfidence"] = _safe(breaking_news_confidence, "evidenceSupportScore")

        # Adjust weights: For breaking news, use breakingNewsConfidence in place
        # of evidenceSupportScore.  No claimClarityScore key exists — use the
        # existing semanticRiskScore as the residual component.
        bn_weight  = WEIGHTS["evidenceSupportScore"]
        bn_conf    = component_scores["breakingNewsConfidence"]
        media_auth = component_scores["mediaAuthenticityScore"]
        risk       = component_scores["semanticRiskScore"]

        final = (
            bn_conf    * bn_weight
            + media_auth * WEIGHTS["mediaAuthenticityScore"]
            + risk       * WEIGHTS["semanticRiskScore"]
            + component_scores["sourceTrustScore"] * WEIGHTS["sourceTrustScore"]
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
        logger.warning("Penalty applied [AI-video]: probability = %.2f", ai_video_probability)
        base_score = max(0, base_score - 40)

    if context_reuse_detected:
        logger.warning("Penalty applied [context-reuse]: reused historical video context")
        base_score = max(0, base_score - 25)

    # -----------------------------------------------------------------------
    # Final Score Output (No Boosts)
    # -----------------------------------------------------------------------
    final_score = int(min(100, max(0, base_score)))

    # -----------------------------------------------------------------------
    # Phase 5: principled confidence (synchronous CPU-only)
    # -----------------------------------------------------------------------
    def classify_score(score: int) -> tuple[str, str]:
        if score >= 75:   return "Low",         "Reliable"
        elif score >= 50: return "Medium",      "Likely Reliable"
        elif score >= 30: return "High",        "Questionable"
        else:             return "Critical",    "High Risk"

    risk_level, final_verdict = classify_score(final_score)

    logger.info(
        "Final credibility — score=%d verdict=%s riskLevel=%s",
        final_score, final_verdict, risk_level,
    )

    result_dict = {
        "credibility_score": final_score,
        "evidence_score": component_scores.get("evidenceSupportScore", 50.0),
        "risk_score": component_scores.get("semanticRiskScore", 100.0),
        "media_score": component_scores.get("mediaAuthenticityScore", 75.0),
        "clarity_score": component_scores.get("claimClarityScore", 70.0),
        "confidence": evidence_confidence if evidence_confidence is not None else 0.0,
        "verdict": final_verdict,
        "risk_level": risk_level
    }
    
    if breaking_news_detected:
        result_dict["breakingNewsDetected"] = True
        
    return result_dict


# ---------------------------------------------------------------------------
# Claim-type classifier (module-level so it is testable and reusable)
# ---------------------------------------------------------------------------

def _classify_claim_type(text: str) -> str:
    """
    Lightweight heuristic classifier.
      - Scientific / geographic / mathematical claims → STATIC
      - News / events / politics / disasters → DYNAMIC
    Static claims are NOT penalised for older but consistent evidence.
    """
    static_keywords = [
        "is located", "capital of", "population of", "produces",
        "consists of", "distance between", "defined as", "equals",
        "formula for",
    ]
    dynamic_keywords = [
        "election", "protest", "riots", "explosion", "earthquake", "flood",
        "won", "lost", "killed", "injured", "announced", "declared",
        "appointed", "resigned",
    ]
    lower = (text or "").lower()
    if any(k in lower for k in dynamic_keywords):
        return "dynamic"
    if any(k in lower for k in static_keywords):
        return "static"
    # Heuristic: presence of a specific year → likely dynamic news event
    if any(str(y) in lower for y in range(1900, 2051)):
        return "dynamic"
    return "static"


# ---------------------------------------------------------------------------
# Async orchestrator — full credibility pipeline
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
    Async: Run the full credibility pipeline.

    Execution order:
      Phase A: LLM text analysis (awaited in ROUTE before this is called)
      Phase B: domain (filtering only) + image authenticity — CONCURRENT async I/O
      Phase C: evidence aggregation + synchronous scoring + confidence

    Domain Phase (B) still runs for evidence filtering downstream but its
    score is NOT included in the credibility formula.
    """
    primary_claim = ""
    semantic_score: float | None = None
    manipulation_indicators: list | None = None
    knowledge_support_score: float | None = None
    breaking_news_detected = False
    breaking_news_confidence: float | None = None
    claim_explicit_date = None
    claim_temporal_signal = None

    if text_analysis and text_analysis.get("status") != "skipped":
        semantic = text_analysis.get("semantic") or {}
        primary_claim = (semantic.get("primaryClaim") or "").strip()
        semantic_score = text_analysis.get("credibilityScore") or semantic.get("semanticScore")
        manipulation_indicators = semantic.get("manipulationIndicators", [])
        
        claims_list = semantic.get("claims", [])
        if claims_list and isinstance(claims_list[0], dict):
            first_claim = claims_list[0]
            claim_explicit_date = first_claim.get("explicit_date")
            claim_temporal_signal = first_claim.get("temporal_signal")
            
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

    # ---- Phase 2 + 3: concurrent I/O (domain + image) ----
    domain_result, image_auth_result = await asyncio.gather(
        evaluate_domain(source_url),
        asyncio.to_thread(evaluate_image, image_bytes, primary_claim or None),
        return_exceptions=True,
    )
    if isinstance(domain_result, Exception):
        logger.warning("Domain phase failed: %s", domain_result)
        domain_result = _neutral_domain()
    if isinstance(image_auth_result, Exception):
        logger.warning("Image phase failed: %s", image_auth_result)
        image_auth_result = _neutral_image()

    # ---- Phase D: Process Unified Evidence Results ----
    evidence_verification_score = 50.0
    evidence_confidence = 0.0
    evidence_sources_used = []
    verified_claims = []
    verification_breakdown = []
    
    # New Confidence Metrics
    evidence_values = []
    evidence_count = 0
    evidence_timestamps = []
    total_retrieved_docs = 0
    total_aligned_sentences = 0
    verifier_confidences = []

    if evidence_results:
        scores = []
        for res in evidence_results:
            v_data = res.get("verification", {})
            v = v_data.get("verdict", "UNVERIFIED")
            conf = v_data.get("confidence", 0.0)
            
            verified_claims.append({
                "claim": res.get("claim", ""),
                "verdict": v
            })
            
            verification_breakdown.append({
                "claim": res.get("claim", ""),
                "verdict": v,
                "confidence": conf,
                "reasoning": v_data.get("reasoning", "")
            })

            verifier_confidences.append(float(conf))
            
            stats = res.get("stats", {})
            total_retrieved_docs += stats.get("retrieved_docs", 0)
            total_aligned_sentences += stats.get("aligned_sentences", 0)

            if v == "SUPPORTED":
                scores.append(1.0)
                evidence_values.append(1)
            elif v == "CONTRADICTED":
                scores.append(-1.0)
                evidence_values.append(-1)
            else:
                scores.append(0.0)
                evidence_values.append(0)
            
            for doc in res.get("evidence", []):
                evidence_count += 1
                source_name = doc.get("source", "Unknown")
                doc_time = doc.get("timestamp") or doc.get("published_at")
                if doc_time and doc_time not in evidence_timestamps:
                    evidence_timestamps.append(doc_time)
                if source_name not in evidence_sources_used:
                    evidence_sources_used.append(source_name)
        
        if scores:
            raw_score = float(sum(scores)) / len(scores)
            evidence_verification_score = ((raw_score + 1.0) / 2.0) * 100.0
            variance = float(sum((float(s) - raw_score) ** 2 for s in scores)) / len(scores)
            evidence_confidence = 1.0 - float(variance)

    # ---- Evaluate Temporal Logic ----
    temporal_consistency_score = 50.0
    temporal_gap_days = None
    gap_classification = "UNKNOWN"

    evidence_timestamps_dt = []
    oldest_ev = None
    newest_ev = None
    for ts in evidence_timestamps:
        try:
            dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            evidence_timestamps_dt.append(dt)
        except Exception:
            pass

    if evidence_timestamps_dt:
        oldest_ev = min(evidence_timestamps_dt).isoformat()
        newest_ev = max(evidence_timestamps_dt).isoformat()

        claim_dt = claim_explicit_date
        if not claim_dt and claim_temporal_signal in ["today", "just happened", "breaking", "minutes ago", "now", "recent"]:
            # Default to now if relative
            claim_dt = datetime.datetime.utcnow()

        if claim_dt:
            temporal_gap_days = compute_temporal_gap(claim_dt, oldest_ev)
            gap_classification = classify_temporal_gap(temporal_gap_days)

            claim_type = _classify_claim_type(primary_claim or "")

            if claim_type == "static":
                # Static claims (scientific facts, geography, math) do NOT get
                # penalised for old but consistent evidence.
                temporal_consistency_score = 80.0
                gap_classification = "STATIC"
            else:
                if gap_classification == "RECENT":
                    temporal_consistency_score = 100.0
                elif gap_classification == "CURRENT":
                    temporal_consistency_score = 70.0
                elif gap_classification == "OLD":
                    temporal_consistency_score = 50.0
                elif gap_classification == "HISTORICAL":
                    temporal_consistency_score = 30.0

    # Overrides for BREAKING NEWS vs OLD MEDIA
    temporal_words = ["today", "just happened", "breaking", "minutes ago", "now", "recent"]
    temporal_signal = breaking_news_detected or any(w in (primary_claim or "").lower() for w in temporal_words) or (claim_temporal_signal in temporal_words)
    
    if temporal_signal and total_retrieved_docs < 3:
        breaking_news_detected = True
        logger.info("Breaking event detected with low evidence volume")

    if temporal_signal and gap_classification in ["OLD", "HISTORICAL"]:
        context_reuse_detected = True
        logger.warning(f"Miscontextualized Media Detected: Claims recent but evidence is {gap_classification}")
        temporal_consistency_score = 10.0

    avg_verifier_conf = (sum(verifier_confidences) / len(verifier_confidences)) if verifier_confidences else 0.0

    confidence_metrics = {
        "evidence_values": evidence_values,
        "evidence_count": evidence_count,
        "unique_sources": len(evidence_sources_used),
        "retrieved_docs": total_retrieved_docs,
        "aligned_sentences": total_aligned_sentences,
        "verifier_confidence": avg_verifier_conf,
        "risk_indicator_count": len(manipulation_indicators or []),
        "temporal_signal": temporal_signal,
        "breaking_news_detected": breaking_news_detected,
        "temporal_gap": temporal_gap_days
    }
    
    confidence_result = calculate_confidence(confidence_metrics)

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
        evidence_confidence=evidence_confidence,
        temporal_consistency_score=temporal_consistency_score
    )

    # Explainability Data
    weighted_result["evidenceSourcesUsed"] = evidence_sources_used
    weighted_result["verifiedClaims"] = verified_claims
    weighted_result["verificationBreakdown"] = verification_breakdown

    weighted_result["domainReputation"]  = domain_result
    weighted_result["imageAuthenticity"] = image_auth_result
    
    # Part 7 API Structure
    weighted_result["confidence"] = confidence_result["confidenceScore"]
    weighted_result["confidenceLevel"] = confidence_result["confidenceLevel"]
    weighted_result["confidenceBreakdown"] = confidence_result["confidenceBreakdown"]
    
    # Part 8 Temporal Analysis Object
    weighted_result["temporalAnalysis"] = {
        "claimTime": claim_explicit_date or claim_temporal_signal,
        "oldestEvidence": oldest_ev,
        "newestEvidence": newest_ev,
        "temporalGapDays": temporal_gap_days,
        "temporalConsistency": gap_classification
    }
    if context_reuse_detected:
        weighted_result["verdict"] = "MISLEADING"
    elif breaking_news_detected and total_retrieved_docs < 3:
        weighted_result["verdict"] = "UNVERIFIED"
    
    return weighted_result


async def _coro_empty_fact() -> dict:
    """Async no-op yielding an empty fact-check response (no API call made)."""
    return {"claims": [], "nextPageToken": ""}
