"""
Agent 3 — Source Credibility Agent

Combines LLM-based credibility assessment with the deterministic
source registry (Part-9) to produce a credibility_index.
The LLM provides narrative labels; the registry provides ground truth.
"""

import logging
from app.config.settings import Config
from app.config.azure import get_async_azure_client
from app.services.verification_agents._utils import extract_json, GLOBAL_SAFETY_RULES
from app.services.evidence_pipeline.source_registry import (
    TIER1_SOURCES, TIER2_SOURCES, TIER3_SOURCES, FACTCHECK_DOMAINS
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""\
{GLOBAL_SAFETY_RULES}

You are a source credibility analyst.

Your task is to evaluate the reliability of source domains in a fact-checking context.

STRICT RULES:
• Only evaluate based on the provided domain names.
• Do not fabricate reputational claims about sources not in your training data.
• If a source is unfamiliar or you are uncertain, classify it as UNKNOWN.
• credibility must be one of: HIGH | MEDIUM | LOW | UNKNOWN
• Do NOT use subjective political opinions about sources.

Credibility guidelines:
  HIGH    — well-established news wire services, government agencies, peer-reviewed journals
  MEDIUM  — established regional outlets, established digital publications
  LOW     — anonymous blogs, known misinformation outlets, tabloids
  UNKNOWN — domain not recognisable
"""

USER_TEMPLATE = """\
Source domains to evaluate:
{domains_list}

Return ONLY valid JSON:
{{
  "source_evaluations": [
    {{
      "domain": "<domain string>",
      "credibility": "HIGH | MEDIUM | LOW | UNKNOWN",
      "reason": "<one sentence>"
    }}
  ],
  "credibility_index": <float 0.0-1.0, average credibility across all sources>
}}
"""

# Deterministic tier → score mapping (fallback / override for known domains)
_TIER_SCORES = {
    1: 1.0,
    2: 0.75,
    3: 0.30,
    0: 0.45,    # unknown
}


def _registry_credibility(domain: str) -> float:
    """Deterministic lookup using Part-9 source registry."""
    if domain in TIER1_SOURCES or domain in FACTCHECK_DOMAINS:
        return _TIER_SCORES[1]
    if domain in TIER2_SOURCES:
        return _TIER_SCORES[2]
    if domain in TIER3_SOURCES:
        return _TIER_SCORES[3]
    return _TIER_SCORES[0]


async def analyze_sources(evidence: list) -> dict:
    """
    Agent 3: Evaluate source credibility.
    evidence: list of dicts with domain field.
    Returns: {"source_evaluations": [...], "credibility_index": float}
    """
    # Extract unique domains
    domains = list({item.get("domain", "unknown") for item in evidence if item.get("domain")})
    _fallback = {
        "source_evaluations": [
            {"domain": d, "credibility": "UNKNOWN", "reason": "Agent unavailable"}
            for d in domains
        ],
        "credibility_index": 0.5,
    }

    if not domains:
        return _fallback

    domains_list = "\n".join(f"- {d}" for d in domains)

    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(domains_list=domains_list)},
            ],
            temperature=0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        result = extract_json(raw)

        # Override credibility_index with deterministic registry score (Part-9 authoritative)
        registry_scores = [_registry_credibility(d) for d in domains]
        result["credibility_index"] = round(
            sum(registry_scores) / len(registry_scores), 3
        ) if registry_scores else 0.5

        logger.info(f"[SourceAgent] {len(domains)} domains, credibility_index={result['credibility_index']}")
        return result

    except Exception as exc:
        logger.warning(f"[SourceAgent] Failed: {exc}")
        # Always return deterministic scores even on LLM failure
        registry_scores = [_registry_credibility(d) for d in domains]
        _fallback["credibility_index"] = round(
            sum(registry_scores) / len(registry_scores), 3
        ) if registry_scores else 0.5
        return _fallback
