"""
Agent 2 — Evidence Validation Agent

Evaluates each evidence sentence independently against the claim,
returning per-evidence SUPPORT / CONTRADICT / NEUTRAL classifications
with confidence scores and short reasoning.
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

You are an evidence evaluation specialist.

Your job is to determine how each evidence item relates to the claim.
Evaluate EVERY evidence item independently.

Possible relationship values:
  SUPPORT    — evidence confirms or is consistent with the claim
  CONTRADICT — evidence refutes or is inconsistent with the claim
  NEUTRAL    — evidence does not address the claim

STRICT RULES:
• Use ONLY the provided evidence text for each evaluation.
• Do not use external knowledge or world models.
• Do not infer facts not explicitly stated.
• If an item is unclear or ambiguous, classify it as NEUTRAL.
• confidence must be a float between 0.0 and 1.0.
• Never attribute information from one source to another.
"""

USER_TEMPLATE = """\
Claim:
{{claim}}

Evidence Items:
{{evidence_block}}

Return ONLY valid JSON:
{{
  "evaluations": [
    {{
      "evidence_id": <int, 1-indexed matching the input IDs>,
      "relation": "SUPPORT | CONTRADICT | NEUTRAL",
      "confidence": <float 0.0-1.0>,
      "reason": "<one concise sentence explaining the classification>"
    }}
  ]
}}
"""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    retry_error_callback=lambda retry_state: logger.warning(f"[EvidenceAgent] Retrying LLM call: attempt {retry_state.attempt_number}")
)
async def evaluate_evidence(
    claim: str, 
    claim_analysis: dict, 
    evidence_items: list,
    request_id: str = "REQ-UNKNOWN"
) -> dict:
    """
    Agent 2: Compare each evidence item against the claim (NLI-style).
    Returns list of evaluations (SUPPORT | CONTRADICT | NEUTRAL).
    """
    _fallback = {"evaluations": [], "explanation": "Failed to evaluate evidence."}
    if not evidence_items:
        return _fallback

    # Build evidence block for the prompt
    lines = []
    for i, item in enumerate(evidence_items):
        eid    = item.get("id", i+1)
        source = item.get("source", "Unknown")
        domain = item.get("domain", "unknown")
        text   = item.get("text", "")
        lines.append(
            f"[{eid}] Source: {source} | Domain: {domain}\n"
            f"     Text: {text}"
        )
    evidence_block = "\n\n".join(lines)

    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    claim=claim,
                    evidence_block=evidence_block
                )},
            ],
            temperature=0,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
        result = extract_json(raw)
        evals = result.get("evaluations", [])
        logger.info(f"[{request_id}] [EvidenceAgent] {len(evals)} evaluations received for {len(evidence_items)} items.")
        return result
    except Exception as exc:
        logger.warning(f"[{request_id}] [EvidenceAgent] Failed: {exc}")
        return _fallback
