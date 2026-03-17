"""
Agent 5 — Consensus Agent

Synthesizes the outputs of all other agents (Metadata, Evidence,
Source, Temporal) into a final verdict, credibility score, and explanation.
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

You are a master fact-checker and consensus builder.

You will be provided with:
1. The original claim.
2. Metadata analysis of the claim.
3. Evidence evaluations (SUPPORT / CONTRADICT / NEUTRAL).
4. Source credibility assessments.
5. Temporal consistency findings.

Your goal is to synthesize these into a single authoritative verdict and a user-friendly explanation.

VERDICT TYPES:
- TRUE: Overwhelmingly supported by high-credibility sources with NO temporal issues.
- MOSTLY_TRUE: Supported by mostly credible sources; minor caveats.
- MIXED: Evidence is contradictory or inconclusive.
- MOSTLY_FALSE: Contradicted by credible sources; minor supporting evidence.
- FALSE: Overwhelmingly contradicted by high-credibility sources.
- UNVERIFIED: Insufficient or extremely low-quality evidence.

EXPLANATION RULES (Write for non-expert users):
1. If the claim is FALSE or MOSTLY_FALSE:
   - Explain clearly WHY it is false (e.g., miscontextualized, fabricated, or debunked).
   - Provide the CORRECT information based on the evidence.
   - Example: "The claim is false because no credible news outlets reported this event. The viral post misrepresents an older video recorded in 2021."

2. If the claim is TRUE or MOSTLY_TRUE:
   - Explain WHY it is accurate (e.g., confirmed by multiple reliable sources).
   - Provide additional helpful context.
   - Example: "The claim is accurate. Multiple news outlets including Reuters and BBC reported this event. The incident occurred on March 3rd during a public rally."

3. For MIXED or UNVERIFIED:
   - State that the claim cannot be fully verified.
   - Mention what parts are supported and what parts are contradictory or missing.

Return:
- verdict
- credibility_score (0-100)
- confidence_score (0.0-1.0)
- explanation (user-friendly synthesis)
"""

USER_TEMPLATE = """\
Claim: {{claim}}

Claim Analysis:
{{claim_analysis}}

Evidence Evaluations:
{{evidence_analysis}}

Source Credibility:
{{source_analysis}}

Temporal Consistency:
{{temporal_analysis}}

Return ONLY valid JSON:
{{
  "verdict": "TRUE | MOSTLY_TRUE | MIXED | MOSTLY_FALSE | FALSE | UNVERIFIED",
  "credibility_score": <int 0-100>,
  "confidence_score": <float 0.0-1.0>,
  "explanation": "<concise paragraph synthesis>"
}}
"""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    retry_error_callback=lambda retry_state: logger.warning(f"[ConsensusAgent] Retrying LLM call: attempt {retry_state.attempt_number}")
)
async def synthesize_verdict(
    claim: str,
    claim_analysis: dict,
    evidence_analysis: dict,
    source_analysis: dict,
    temporal_analysis: dict,
    request_id: str = "REQ-UNKNOWN"
) -> dict:
    """
    Agent 5: Final synthesis of all inputs into a verdict and score.
    Returns: verdict, credibility_score, explanation, confidence_score.
    """
    _fallback = {
        "verdict": "UNVERIFIED",
        "credibility_score": 50,
        "explanation": "Synthesis failed.",
        "confidence_score": 0.5
    }

    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    claim=claim,
                    claim_analysis=json.dumps(claim_analysis, indent=2),
                    evidence_analysis=json.dumps(evidence_analysis, indent=2),
                    source_analysis=json.dumps(source_analysis, indent=2),
                    temporal_analysis=json.dumps(temporal_analysis, indent=2),
                )},
            ],
            temperature=0,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        result = extract_json(raw)
        logger.info(f"[{request_id}] [ConsensusAgent] Verdict: {result.get('verdict')} Score: {result.get('credibility_score')}")
        return result
    except Exception as exc:
        logger.warning(f"[{request_id}] [ConsensusAgent] Failed: {exc}")
        return _fallback
