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

Evidence Items ({{item_count}} total):
{{evidence_block}}

CRITICAL INSTRUCTION: You MUST evaluate EVERY evidence item listed above.
You have {{item_count}} evidence items, so you MUST return exactly {{item_count}} evaluations.
Do NOT stop after the first evaluation. Complete ALL {{item_count}} evaluations.

Return ONLY valid JSON with ALL {{item_count}} evaluations:
{{
  "evaluations": [
    {{
      "evidence_id": 1,
      "relation": "SUPPORT | CONTRADICT | NEUTRAL",
      "confidence": 0.8,
      "reason": "<concise sentence>"
    }},
    {{
      "evidence_id": 2,
      "relation": "SUPPORT | CONTRADICT | NEUTRAL",
      "confidence": 0.7,
      "reason": "<concise sentence>"
    }},
    {{
      "evidence_id": 3,
      "relation": "SUPPORT | CONTRADICT | NEUTRAL",
      "confidence": 0.9,
      "reason": "<concise sentence>"
    }}
    ... (continue for ALL {{item_count}} items)
  ]
}}
"""

async def _evaluate_batch(
    claim: str,
    claim_analysis: dict,
    evidence_items: list,
    request_id: str = "REQ-UNKNOWN",
    max_retries: int = 3
) -> dict:
    """
    Internal helper: Evaluate a single batch of evidence items with exponential backoff.
    """
    _fallback = {"evaluations": []}
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

    # Build full prompt for length diagnosis
    user_content = USER_TEMPLATE.format(
        claim=claim,
        evidence_block=evidence_block,
        item_count=len(evidence_items)
    )

    # Retry loop with exponential backoff for connection errors and rate limiting
    for attempt in range(max_retries):
        try:
            client = get_async_azure_client()
            
            response = await client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0,
                max_tokens=2000,
            )
            
            # Log token usage to diagnose truncation
            usage = response.usage
            logger.info(
                f"[{request_id}] [EvidenceAgent] Token usage: "
                f"prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} "
                f"total={usage.total_tokens} "
                f"(max_tokens=2000, prompt_chars={len(user_content)})"
            )
            
            raw = response.choices[0].message.content.strip()
            result = extract_json(raw)
            evals = result.get("evaluations", [])
            
            # Detect truncation
            if len(evals) < len(evidence_items):
                logger.warning(
                    f"[{request_id}] [EvidenceAgent] Truncation detected: got {len(evals)} evaluations "
                    f"for {len(evidence_items)} items. Raw length: {len(raw)} chars. "
                    f"Completion tokens: {usage.completion_tokens}/2000"
                )
            
            return result
            
        except Exception as exc:
            error_msg = str(exc).lower()
            is_retryable = (
                "connection error" in error_msg or
                "rate limit" in error_msg or
                "429" in error_msg or
                "timeout" in error_msg
            )
            
            if is_retryable and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    f"[{request_id}] [EvidenceAgent] Retryable error (attempt {attempt + 1}/{max_retries}): {exc}. "
                    f"Waiting {wait_time}s before retry..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.warning(f"[{request_id}] [EvidenceAgent] Batch evaluation failed: {exc}")
                return _fallback
    
    return _fallback


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
    Splits large batches (>3 items) to prevent JSON truncation.
    """
    _fallback = {"evaluations": [], "explanation": "Failed to evaluate evidence."}
    if not evidence_items:
        return _fallback

    MAX_ITEMS_PER_CALL = 3
    
    if len(evidence_items) > MAX_ITEMS_PER_CALL:
        # Split into batches and merge results
        logger.info(f"[{request_id}] [EvidenceAgent] Splitting {len(evidence_items)} items into batches of {MAX_ITEMS_PER_CALL}")
        all_evals = []
        for i in range(0, len(evidence_items), MAX_ITEMS_PER_CALL):
            batch = evidence_items[i:i + MAX_ITEMS_PER_CALL]
            batch_result = await _evaluate_batch(claim, claim_analysis, batch, request_id)
            all_evals.extend(batch_result.get("evaluations", []))
        
        logger.info(f"[{request_id}] [EvidenceAgent] {len(all_evals)} evaluations received for {len(evidence_items)} items.")
        return {"evaluations": all_evals}
    else:
        # Original single call for small batches
        result = await _evaluate_batch(claim, claim_analysis, evidence_items, request_id)
        logger.info(f"[{request_id}] [EvidenceAgent] {len(result.get('evaluations', []))} evaluations received for {len(evidence_items)} items.")
        return result
