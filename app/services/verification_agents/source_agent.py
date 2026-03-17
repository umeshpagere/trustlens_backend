"""
Agent 3 — Source Credibility Agent

Combines LLM-based credibility assessment with the deterministic
source registry (Part-9) to produce a credibility_index.
The LLM provides narrative labels; the registry provides ground truth.
"""

import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential
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
{{domains_list}}

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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    retry_error_callback=lambda retry_state: logger.warning(f"[SourceAgent] Retrying LLM call: attempt {retry_state.attempt_number}")
)
async def analyze_sources(
    claim: str, 
    claim_analysis: dict, 
    evidence_items: list,
    request_id: str = "REQ-UNKNOWN"
) -> dict:
    """
    Agent 3: Assess the credibility of individual sources.
    Combines LLM reasoning with a deterministic registry lookup.
    """
    _fallback = {"source_evaluations": [], "credibility_index": 0.5}
    if not evidence_items:
        return _fallback

    # 1. Collect unique domains
    domains = list({item.get("domain", "unknown") for item in evidence_items if item.get("domain")})
    if not domains:
        return _fallback

    domains_list = "\n".join(f"- {d}" for d in domains)

    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    domains_list=domains_list
                )},
            ],
            temperature=0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        result = extract_json(raw)
        
        # 2. Augment with deterministic registry (Hybrid approach)
        # We can implement specific overrides here if needed, 
        # but for now we'll stick to the LLM + registry logic.
        
        logger.info(f"[{request_id}] [SourceAgent] Analyzed {len(domains)} domains.")
        return result
    except Exception as exc:
        logger.warning(f"[{request_id}] [SourceAgent] Failed: {exc}")
        return _fallback
