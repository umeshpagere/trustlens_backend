"""
TrustLens Response Builder Layer

Assembles intermediate pipeline outputs into a unified, structured JSON response
compatible with both the Mobile App and Chrome Extension.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def build_analysis_response(
    credibility_result: Dict[str, Any],
    text_result: Dict[str, Any],
    media_result: Dict[str, Any],
    ai_result: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constructs the final structured API response.
    """
    return {
        "success": True,
        "credibility": {
            "score": credibility_result.get("score", 50),
            "verdict": credibility_result.get("verdict", "UNVERIFIED"),
            "riskLevel": credibility_result.get("riskLevel", "medium"),
            "summary": credibility_result.get("summary", "")
        },
        "textAnalysis": {
            "score": text_result.get("score", 50),
            "verdict": text_result.get("verdict", "UNVERIFIED"),
            "primaryClaim": text_result.get("primaryClaim", ""),
            "analysis": text_result.get("analysis", ""),
            "claims": text_result.get("claims", []),
            "evidenceSources": text_result.get("evidenceSources", []),
            "detailedExplanation": text_result.get("detailedExplanation", "")
        },
        "mediaAnalysis": {
            "score": media_result.get("score", 50),
            "verdict": media_result.get("verdict", "UNVERIFIED"),
            "analysis": media_result.get("analysis", ""),
            "mediaType": media_result.get("mediaType", "unknown"),
            "visualDescription": media_result.get("visualDescription", ""),
            "verification": media_result.get("verification", ""),
            "evidenceSources": media_result.get("evidenceSources", [])
        },
        "aiProbability": {
            "score": ai_result.get("score", 0),
            "verdict": ai_result.get("verdict", "Unknown"),
            "analysis": ai_result.get("analysis", "")
        },
        "meta": {
            "processingTimeMs": meta.get("processingTimeMs", 0),
            "claimsAnalyzed": meta.get("claimsAnalyzed", 0),
            "sourcesUsed": meta.get("sourcesUsed", 0)
        }
    }
