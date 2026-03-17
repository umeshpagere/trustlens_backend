import math
import statistics

def calculate_confidence(metrics: dict) -> dict:
    # Extract inputs
    evidence_values = metrics.get('evidence_values', [])
    evidence_count = metrics.get('evidence_count', 0)
    unique_sources = metrics.get('unique_sources', 0)
    retrieved_docs = metrics.get('retrieved_docs', 0)
    aligned_sentences = metrics.get('aligned_sentences', 0)
    verifier_confidence = metrics.get('verifier_confidence', 0.0)
    risk_indicator_count = metrics.get('risk_indicator_count', 0)
    temporal_signal = metrics.get('temporal_signal', False)
    breaking_news_detected = metrics.get('breaking_news_detected', False)
    temporal_gap = metrics.get('temporal_gap')

    # 1. Evidence Agreement
    if len(evidence_values) > 1:
        variance = statistics.pvariance(evidence_values)
        evidence_agreement = max(0.0, 1.0 - variance)
    elif len(evidence_values) == 1:
        evidence_agreement = 1.0
    else:
        evidence_agreement = 0.0

    # 2. Source Diversity
    source_diversity = min(unique_sources / max(1, evidence_count), 1.0)

    # 3. Evidence Volume
    evidence_volume = min(evidence_count / 10.0, 1.0)

    # 4. Retrieval Coverage
    retrieval_coverage = min(aligned_sentences / max(1, retrieved_docs), 1.0)
    retrieval_coverage = max(0.0, retrieval_coverage)

    # 5. Risk Penalty
    risk_penalty = math.exp(-0.4 * risk_indicator_count)

    # 6. Temporal Uncertainty
    if retrieved_docs < 3 and temporal_signal:
        temporal_penalty = 0.8
    else:
        temporal_penalty = 1.0

    # 7. Final Confidence Formula
    confidence = (
        0.30 * evidence_agreement +
        0.20 * source_diversity +
        0.15 * evidence_volume +
        0.15 * verifier_confidence +
        0.10 * retrieval_coverage +
        0.10 * risk_penalty
    ) * temporal_penalty

    if breaking_news_detected and retrieved_docs < 3:
        confidence -= 0.25

    if temporal_gap is not None and temporal_gap > 365:
        confidence -= 0.30

    # Clamp
    confidence = max(0.05, min(1.0, confidence))

    # 8. Confidence Level Mapping
    if confidence >= 0.85:
        confidence_level = "Very High"
    elif confidence >= 0.70:
        confidence_level = "High"
    elif confidence >= 0.50:
        confidence_level = "Medium"
    elif confidence >= 0.30:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"

    return {
        "confidenceScore": float(round(confidence, 4)),
        "confidenceLevel": confidence_level,
        "confidenceBreakdown": {
            "evidenceAgreement": float(round(evidence_agreement, 4)),
            "sourceDiversity": float(round(source_diversity, 4)),
            "evidenceVolume": float(round(evidence_volume, 4)),
            "verifierConfidence": float(round(verifier_confidence, 4)),
            "retrievalCoverage": float(round(retrieval_coverage, 4)),
            "riskPenalty": float(round(risk_penalty, 4)),
            "temporalPenalty": float(round(temporal_penalty, 4))
        }
    }
