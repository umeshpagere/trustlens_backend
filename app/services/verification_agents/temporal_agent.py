"""
Agent 4 — Temporal Consistency Agent

Compares claim timing signals against evidence publication timestamps
to detect outdated, miscontextualised, or anachronistic evidence.
"""

import logging
from app.config.settings import Config
from app.config.azure import get_async_azure_client
from app.services.verification_agents._utils import extract_json, GLOBAL_SAFETY_RULES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""\
{GLOBAL_SAFETY_RULES}

You are a temporal consistency analyst.

Your task is to evaluate whether the evidence is temporally consistent with the claim.

STRICT RULES:
• Only use the timestamps explicitly provided.
• Do not infer or guess missing dates.
• If a timestamp is missing, classify that item as UNKNOWN.
• Consider the claim temporal signal when determining consistency.

Classification values:
  RECENT    — evidence is recent and consistent with the claim's timeframe
  OUTDATED  — evidence is older and may no longer support the claim
  MISMATCHED — evidence timestamp contradicts the claim's stated timeframe
  UNKNOWN   — no timestamp available for evaluation

temporal_consistency_score: float 0.0–1.0 (1.0 = all evidence is temporally consistent)
"""

USER_TEMPLATE = """\
Claim:
{claim}

Claim Temporal Signal: {temporal_signal}
Claim Explicit Date: {explicit_date}

Evidence Items (with timestamps):
{evidence_block}

Return ONLY valid JSON:
{{
  "temporal_assessment": [
    {{
      "evidence_id": <int, 1-indexed>,
      "classification": "RECENT | OUTDATED | MISMATCHED | UNKNOWN",
      "evidence_date": "<date or UNKNOWN>",
      "reason": "<one sentence>"
    }}
  ],
  "temporal_consistency_score": <float 0.0-1.0>
}}
"""


async def check_temporal_consistency(
    claim: str,
    evidence: list,
    temporal_signal: str = "UNKNOWN",
    explicit_date: str = "UNKNOWN",
) -> dict:
    """
    Agent 4: Check each evidence item's temporal consistency with the claim.
    evidence: list of dicts with id, published_at, text fields.
    Returns: {"temporal_assessment": [...], "temporal_consistency_score": float}
    """
    _fallback = {
        "temporal_assessment": [
            {
                "evidence_id": item.get("id", i + 1),
                "classification": "UNKNOWN",
                "evidence_date": item.get("published_at", "UNKNOWN"),
                "reason": "Temporal agent unavailable",
            }
            for i, item in enumerate(evidence)
        ],
        "temporal_consistency_score": 0.5,
    }

    if not evidence or not claim:
        return _fallback

    lines = []
    for item in evidence:
        eid      = item.get("id", "?")
        pub_at   = item.get("published_at", "UNKNOWN")
        text_snip = item.get("text", "")[:120]
        lines.append(f"[{eid}] Published: {pub_at} | Text: {text_snip}…")
    evidence_block = "\n".join(lines)

    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    claim=claim,
                    temporal_signal=temporal_signal or "UNKNOWN",
                    explicit_date=explicit_date or "UNKNOWN",
                    evidence_block=evidence_block,
                )},
            ],
            temperature=0,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        result = extract_json(raw)

        # Clamp score
        score = result.get("temporal_consistency_score", 0.5)
        result["temporal_consistency_score"] = max(0.0, min(1.0, float(score)))

        logger.info(
            f"[TemporalAgent] temporal_consistency_score={result['temporal_consistency_score']}"
        )
        return result

    except Exception as exc:
        logger.warning(f"[TemporalAgent] Failed: {exc}")
        return _fallback
