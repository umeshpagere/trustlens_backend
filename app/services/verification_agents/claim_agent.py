"""
Agent 1 — Claim Analysis Agent

Extracts structural metadata from the claim WITHOUT verifying it.
Runs first so downstream agents can use its entity/temporal output.
"""

import json
import logging
from app.config.settings import Config
from app.config.azure import get_async_azure_client
from app.services.verification_agents._utils import extract_json, GLOBAL_SAFETY_RULES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""\
{GLOBAL_SAFETY_RULES}

You are a factual claim analysis specialist.

Your job is to analyse the STRUCTURE of a claim, NOT to verify its truth.

STRICT RULES:
• Only extract entities explicitly present in the claim text.
• Do not infer hidden actors or unexpressed context.
• Do not expand abbreviations unless written in full in the claim.
• Do not invent context.
• If information is absent, return "UNKNOWN" for that field.
• claim_complexity must be one of: SIMPLE | COMPLEX | COMPOUND
• expected_evidence_types is a list from: news | fact_check | government | academic | social_media
"""

USER_TEMPLATE = """\
Claim:
{claim}

Return ONLY valid JSON:
{{
  "entities": ["<list of named entities in the claim>"],
  "event_type": "<e.g. political, military, scientific, economic, health>",
  "temporal_signal": "<e.g. today, yesterday, last year, or UNKNOWN>",
  "explicit_date": "<ISO date string if present, else UNKNOWN>",
  "claim_complexity": "SIMPLE | COMPLEX | COMPOUND",
  "expected_evidence_types": ["<news|fact_check|government|academic|social_media>"]
}}
"""


async def analyze_claim(claim: str) -> dict:
    """
    Agent 1: Extract structured metadata from claim text.
    Returns a dict with entities, event_type, temporal_signal, etc.
    """
    _fallback = {
        "entities": [],
        "event_type": "UNKNOWN",
        "temporal_signal": "UNKNOWN",
        "explicit_date": "UNKNOWN",
        "claim_complexity": "SIMPLE",
        "expected_evidence_types": ["news"],
    }

    if not claim or not claim.strip():
        return _fallback

    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(claim=claim)},
            ],
            temperature=0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        result = extract_json(raw)
        logger.info(f"[ClaimAgent] entities={result.get('entities')} signal={result.get('temporal_signal')}")
        return {**_fallback, **result}
    except Exception as exc:
        logger.warning(f"[ClaimAgent] Failed: {exc}")
        return _fallback
