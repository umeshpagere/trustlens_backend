"""
Agent 5 — Consensus Synthesizer

Receives structured outputs from all four upstream agents and
synthesises a final verdict. It does NOT re-read raw evidence.
"""

import logging
from app.config.settings import Config
from app.config.azure import get_async_azure_client
from app.services.verification_agents._utils import extract_json, GLOBAL_SAFETY_RULES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""\
{GLOBAL_SAFETY_RULES}

You are the verification consensus engine for a fact-checking system.

You will receive structured outputs from four specialist agents:
  1. Claim Analysis Agent     — claim structure and entities
  2. Evidence Validation Agent — per-evidence SUPPORT / CONTRADICT / NEUTRAL labels
  3. Source Credibility Agent  — domain credibility index
  4. Temporal Consistency Agent — temporal consistency of evidence

Your task is to synthesise these into a final verification verdict.

STRICT RULES:
• Do NOT re-interpret raw evidence — only use the agent outputs provided.
• If evidence_validation shows strong contradictions, lean towards CONTRADICTED.
• If temporal_consistency_score < 0.40, add a temporal warning note to reasoning.
• If credibility_index < 0.45, reduce confidence by assigning it closer to 0.3.
• If evidence is insufficient or conflicting, return UNVERIFIED.
• credibility_score must reflect evidence strength, NOT domain reputation alone.
• confidence must be a float between 0.0 and 1.0.
• verdict must be exactly one of: SUPPORTED | CONTRADICTED | UNVERIFIED
"""

USER_TEMPLATE = """\
Claim:
{claim}

Claim Analysis:
{claim_analysis}

Evidence Validation:
{evidence_analysis}

Source Credibility:
{source_analysis}

Temporal Consistency:
{temporal_analysis}

Return ONLY valid JSON:
{{
  "verdict": "SUPPORTED | CONTRADICTED | UNVERIFIED",
  "credibility_score": <int 0-100>,
  "confidence": <float 0.0-1.0>,
  "support_count": <int>,
  "contradict_count": <int>,
  "temporal_warning": "<string or null>",
  "reasoning": "<detailed explanation referencing agent outputs>"
}}
"""


async def synthesize_verdict(
    claim: str,
    claim_analysis: dict,
    evidence_analysis: dict,
    source_analysis: dict,
    temporal_analysis: dict,
) -> dict:
    """
    Agent 5: Synthesize final verdict from all agent outputs.
    Returns a dict compatible with the existing verification result schema.
    """
    _fallback = {
        "verdict": "UNVERIFIED",
        "credibility_score": 50,
        "confidence": 0.3,
        "support_count": 0,
        "contradict_count": 0,
        "temporal_warning": None,
        "reasoning": "Consensus agent unavailable.",
    }

    import json as _json

    def _safe_json(obj: dict) -> str:
        try:
            return _json.dumps(obj, indent=2)
        except Exception:
            return str(obj)

    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    claim=claim,
                    claim_analysis=_safe_json(claim_analysis),
                    evidence_analysis=_safe_json(evidence_analysis),
                    source_analysis=_safe_json(source_analysis),
                    temporal_analysis=_safe_json(temporal_analysis),
                )},
            ],
            temperature=0,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        result = extract_json(raw)

        # Clamp numeric fields
        result["confidence"]       = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        result["credibility_score"] = max(0, min(100, int(result.get("credibility_score", 50))))

        # Normalise verdict
        verdict_map = {
            "SUPPORTED": "SUPPORTED", "SUPPORT": "SUPPORTED", "TRUE": "SUPPORTED",
            "CONTRADICTED": "CONTRADICTED", "CONTRADICT": "CONTRADICTED", "FALSE": "CONTRADICTED",
        }
        raw_verdict = str(result.get("verdict", "UNVERIFIED")).upper()
        result["verdict"] = verdict_map.get(raw_verdict, "UNVERIFIED")

        logger.info(
            f"[ConsensusAgent] verdict={result['verdict']} "
            f"score={result['credibility_score']} confidence={result['confidence']}"
        )
        return result

    except Exception as exc:
        logger.warning(f"[ConsensusAgent] Failed: {exc}")
        return _fallback
