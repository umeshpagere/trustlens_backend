"""
Agent 4 — Temporal Consistency Agent

Detects if the claim is outdated compared to the evidence, or if the
evidence follows a chronological sequence that refutes a static claim.
"""

import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config.settings import Config
from app.config.azure import get_async_azure_client
from app.services.verification_agents._utils import extract_json, GLOBAL_SAFETY_RULES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""\
{GLOBAL_SAFETY_RULES}

You are a temporal consistency expert.

Your job is to identify contradictions between the claim's stated time/event status and the timestamps or sequence of the evidence.

Key Checks:
1. Is the claim "current" but the evidence shows a more recent change?
2. Does the claim specify a date that contradicts the evidence publication dates?
3. Is the evidence too old to support a claim about a recent event?

Return:
- temporal_consistency_score (0.0 to 1.0)
- temporal_assessment (list of issues found)
"""

USER_TEMPLATE = """\
Claim:
{{claim}}

Claim Temporal Signal: {{claim_temporal_signal}}

Evidence with Timestamps:
{{evidence_block}}

Return ONLY valid JSON:
{{
  "temporal_consistency_score": <float 0.0-1.0>,
  "temporal_assessment": [
    {{
      "issue": "<concise description>",
      "severity": "HIGH | MEDIUM | LOW",
      "evidence_id": <int or null>
    }}
  ]
}}
"""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    retry_error_callback=lambda retry_state: logger.warning(f"[TemporalAgent] Retrying LLM call: attempt {retry_state.attempt_number}")
)
async def check_temporal_consistency(
    claim: str, 
    claim_analysis: dict, 
    evidence_items: list,
    temporal_signal: str = None,
    explicit_date: str = None,
    request_id: str = "REQ-UNKNOWN"
) -> dict:
    """
    Agent 4: Detect temporal contradictions or stale information.
    """
    _fallback = {"temporal_assessment": [], "temporal_consistency_score": 0.5}
    if not evidence_items:
        return _fallback

    # Filter items with timestamps
    timed_items = []
    for i, item in enumerate(evidence_items):
        if item.get("published_at"):
            timed_items.append({
                "id": item.get("id", i+1),
                "text": item.get("text", "")[:200] + "...",
                "published_at": item.get("published_at")
            })

    if not timed_items:
        return {"temporal_assessment": [], "temporal_consistency_score": 0.5, "reasoning": "No evidence timestamps found."}

    evidence_block = json.dumps(timed_items, indent=2)

    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    claim=claim,
                    claim_temporal_signal=temporal_signal or claim_analysis.get("temporal_signal", "UNKNOWN"),
                    evidence_block=evidence_block
                )},
            ],
            temperature=0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        result = extract_json(raw)
        logger.info(f"[{request_id}] [TemporalAgent] Analyzed {len(timed_items)} timed items. Score: {result.get('temporal_consistency_score')}")
        return result
    except Exception as exc:
        logger.warning(f"[{request_id}] [TemporalAgent] Failed: {exc}")
        return _fallback
