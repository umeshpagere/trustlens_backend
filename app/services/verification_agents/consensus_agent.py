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

IMPORTANT: The NLI Pre-filter Summary shows results from an automated
Natural Language Inference model that ran BEFORE LLM evaluation.

Use it as a strong signal:
- If NLI shows 3+ contradicting and 0 supporting → strong evidence of FALSE
- If NLI shows 3+ supporting and 0 contradicting → strong evidence of TRUE
- If NLI is mixed → rely more on LLM evaluations

CRITICAL: DO NOT DEFAULT TO UNVERIFIED IF EVIDENCE EXISTS.

VERDICT DECISION TREE (follow strictly in order):

Step 1: Check NLI Pre-filter + LLM Evaluations Together
- If NLI supporting >= 3 AND any LLM evaluation shows SUPPORT:
  → verdict=SUPPORTED or MOSTLY_TRUE, score=70-85
  
- If NLI contradicting >= 3 AND any LLM evaluation shows CONTRADICT:
  → verdict=CONTRADICTED or MOSTLY_FALSE, score=15-35

Step 2: Check LLM Evaluations Alone (if NLI < 3)
- If 3+ evaluations are SUPPORT with credibility_index > 0.5:
  → verdict=TRUE or MOSTLY_TRUE, score=70-85
  
- If 2+ evaluations are SUPPORT and 0 CONTRADICT:
  → verdict=MOSTLY_TRUE, score=60-75
  
- If 3+ evaluations are CONTRADICT:
  → verdict=FALSE or MOSTLY_FALSE, score=15-35

Step 3: Mixed Evidence
- If both SUPPORT and CONTRADICT evaluations exist:
  → verdict=MIXED, score=40-55

Step 4: UNVERIFIED (ONLY if truly no evidence)
- ONLY use UNVERIFIED if:
  * NLI supporting = 0 AND NLI contradicting = 0
  * AND all evaluations are NEUTRAL or empty
  * AND credibility_index < 0.3
  → verdict=UNVERIFIED, score=45-55

EXAMPLES:

Example 1 (SUPPORTED):
NLI: 4 supporting, 0 contradicting
Evaluations: [{{"relation": "SUPPORT", "confidence": 0.8}}, {{"relation": "SUPPORT", "confidence": 0.7}}]
→ Verdict: SUPPORTED, Score: 75

Example 2 (CONTRADICTED):
NLI: 0 supporting, 4 contradicting
Evaluations: [{{"relation": "CONTRADICT", "confidence": 0.9}}]
→ Verdict: CONTRADICTED, Score: 25

Example 3 (MIXED):
NLI: 2 supporting, 2 contradicting
Evaluations: [{{"relation": "SUPPORT"}}, {{"relation": "CONTRADICT"}}]
→ Verdict: MIXED, Score: 50

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

NLI Pre-filter Summary:
{{nli_summary}}

Evidence Evaluations:
{{evidence_analysis}}

Source Credibility:
{{source_analysis}}

Temporal Consistency:
{{temporal_analysis}}

Return ONLY valid JSON:
{{
  "verdict": "SUPPORTED | LIKELY_SUPPORTED | CONTRADICTED | LIKELY_CONTRADICTED | TRUE | MOSTLY_TRUE | MIXED | MOSTLY_FALSE | FALSE | UNVERIFIED",
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
    nli_summary: str = "",
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

    # Log input to diagnose UNVERIFIED issue
    logger.info(
        f"[{request_id}] [ConsensusAgent] Input — evaluations={len(evidence_analysis.get('evaluations', []))} "
        f"credibility_index={source_analysis.get('credibility_index', 'N/A')} nli_summary='{nli_summary}'"
    )
    
    # Diagnostic: Log actual evidence_analysis structure
    logger.debug(f"[{request_id}] [ConsensusAgent] Evidence Analysis JSON: {json.dumps(evidence_analysis, indent=2)}")
    
    # Count SUPPORT/CONTRADICT/NEUTRAL in evaluations
    evaluations = evidence_analysis.get('evaluations', [])
    support_count = sum(1 for e in evaluations if e.get('relation') == 'SUPPORT')
    contradict_count = sum(1 for e in evaluations if e.get('relation') == 'CONTRADICT')
    neutral_count = sum(1 for e in evaluations if e.get('relation') == 'NEUTRAL')
    logger.info(
        f"[{request_id}] [ConsensusAgent] Evaluation breakdown: "
        f"{support_count} SUPPORT, {contradict_count} CONTRADICT, {neutral_count} NEUTRAL"
    )

    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    claim=claim,
                    claim_analysis=json.dumps(claim_analysis, indent=2),
                    nli_summary=nli_summary or "No NLI pre-filter data available",
                    evidence_analysis=json.dumps(evidence_analysis, indent=2),
                    source_analysis=json.dumps(source_analysis, indent=2),
                    temporal_analysis=json.dumps(temporal_analysis, indent=2),
                )},
            ],
            temperature=0,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
        logger.debug(f"[{request_id}] [ConsensusAgent] Raw LLM response: {raw}")
        result = extract_json(raw)
        
        # Programmatic fallback: Override UNVERIFIED when clear evidence exists
        # This prevents LLM from incorrectly defaulting to UNVERIFIED
        if result.get('verdict') == 'UNVERIFIED':
            # Parse NLI summary to get counts
            nli_supporting = 0
            nli_contradicting = 0
            if nli_summary:
                import re
                match = re.search(r'(\d+)\s+supporting,\s+(\d+)\s+contradicting', nli_summary)
                if match:
                    nli_supporting = int(match.group(1))
                    nli_contradicting = int(match.group(2))
            
            # Override logic based on clear evidence patterns
            if nli_supporting >= 3 and support_count >= 1 and contradict_count == 0:
                logger.warning(
                    f"[{request_id}] [ConsensusAgent] Overriding UNVERIFIED → SUPPORTED "
                    f"(NLI: {nli_supporting} supporting, Evals: {support_count} SUPPORT)"
                )
                result['verdict'] = 'SUPPORTED'
                result['credibility_score'] = 75
                result['explanation'] = (
                    f"The claim is supported by evidence. {nli_supporting} supporting sentences were found "
                    f"and {support_count} evidence evaluations confirmed support with no contradictions."
                )
            elif nli_contradicting >= 3 and contradict_count >= 1 and support_count == 0:
                logger.warning(
                    f"[{request_id}] [ConsensusAgent] Overriding UNVERIFIED → CONTRADICTED "
                    f"(NLI: {nli_contradicting} contradicting, Evals: {contradict_count} CONTRADICT)"
                )
                result['verdict'] = 'CONTRADICTED'
                result['credibility_score'] = 25
                result['explanation'] = (
                    f"The claim is contradicted by evidence. {nli_contradicting} contradicting sentences were found "
                    f"and {contradict_count} evidence evaluations confirmed contradictions."
                )
            elif support_count >= 2 and contradict_count == 0:
                logger.warning(
                    f"[{request_id}] [ConsensusAgent] Overriding UNVERIFIED → MOSTLY_TRUE "
                    f"(Evals: {support_count} SUPPORT, 0 CONTRADICT)"
                )
                result['verdict'] = 'MOSTLY_TRUE'
                result['credibility_score'] = 65
                result['explanation'] = (
                    f"The claim is mostly true based on {support_count} supporting evidence evaluations "
                    f"with no contradictions found."
                )
            elif support_count >= 1 and contradict_count >= 1:
                logger.warning(
                    f"[{request_id}] [ConsensusAgent] Overriding UNVERIFIED → MIXED "
                    f"(Evals: {support_count} SUPPORT, {contradict_count} CONTRADICT)"
                )
                result['verdict'] = 'MIXED'
                result['credibility_score'] = 50
                result['explanation'] = (
                    f"The claim has mixed evidence with {support_count} supporting and "
                    f"{contradict_count} contradicting evaluations."
                )
            elif support_count == 0 and contradict_count == 0 and neutral_count >= 1:
                # All evaluations are NEUTRAL - use NLI as tiebreaker
                if nli_supporting >= 2 and nli_contradicting >= 1:
                    logger.warning(
                        f"[{request_id}] [ConsensusAgent] Overriding UNVERIFIED → MIXED "
                        f"(NLI: {nli_supporting} supporting, {nli_contradicting} contradicting, Evals: all NEUTRAL)"
                    )
                    result['verdict'] = 'MIXED'
                    result['credibility_score'] = 50
                    result['explanation'] = (
                        f"The claim has mixed evidence. NLI analysis found {nli_supporting} supporting "
                        f"and {nli_contradicting} contradicting sentences, though detailed evaluation "
                        f"found the evidence inconclusive."
                    )
                elif nli_supporting >= 3 and nli_contradicting == 0:
                    logger.warning(
                        f"[{request_id}] [ConsensusAgent] Overriding UNVERIFIED → LIKELY_SUPPORTED "
                        f"(NLI: {nli_supporting} supporting, 0 contradicting, Evals: all NEUTRAL)"
                    )
                    result['verdict'] = 'LIKELY_SUPPORTED'
                    result['credibility_score'] = 65
                    result['explanation'] = (
                        f"The claim is likely supported. NLI analysis found {nli_supporting} supporting "
                        f"sentences with no contradictions."
                    )
                elif nli_contradicting >= 3 and nli_supporting == 0:
                    logger.warning(
                        f"[{request_id}] [ConsensusAgent] Overriding UNVERIFIED → LIKELY_CONTRADICTED "
                        f"(NLI: {nli_contradicting} contradicting, 0 supporting, Evals: all NEUTRAL)"
                    )
                    result['verdict'] = 'LIKELY_CONTRADICTED'
                    result['credibility_score'] = 35
                    result['explanation'] = (
                        f"The claim is likely contradicted. NLI analysis found {nli_contradicting} "
                        f"contradicting sentences with no support."
                    )
                elif nli_supporting >= 2 and nli_contradicting == 0:
                    logger.warning(
                        f"[{request_id}] [ConsensusAgent] Overriding UNVERIFIED → MOSTLY_TRUE "
                        f"(NLI: {nli_supporting} supporting, 0 contradicting, Evals: all NEUTRAL)"
                    )
                    result['verdict'] = 'MOSTLY_TRUE'
                    result['credibility_score'] = 60
                    result['explanation'] = (
                        f"The claim is mostly true. NLI analysis found {nli_supporting} supporting "
                        f"sentences with no contradictions."
                    )
        
        logger.info(f"[{request_id}] [ConsensusAgent] Verdict: {result.get('verdict')} Score: {result.get('credibility_score')}")
        logger.debug(f"[{request_id}] [ConsensusAgent] Full result: {json.dumps(result, indent=2)}")
        return result
    except Exception as exc:
        logger.warning(f"[{request_id}] [ConsensusAgent] Failed: {exc}")
        return _fallback
