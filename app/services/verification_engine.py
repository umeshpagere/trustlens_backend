"""
TrustLens Verification Engine — Part-11 (Multi-Agent)

verify_claim_credibility() is now a thin shim that delegates to the
multi-agent orchestrator.  All original parameters are preserved for
backward compatibility with pipeline.py.

The monolithic single-LLM pattern is replaced by:
  Phase A: Claim Analysis Agent
  Phase B: Evidence + Source + Temporal agents (parallel)
  Phase C: Consensus Synthesizer

Output schema is identical to Part-10 so credibility_engine and
confidence_service require zero changes.
"""

import logging
from app.services.verification_agents.orchestrator import run_verification_agents

logger = logging.getLogger(__name__)


async def verify_claim_credibility(
    claim: str,
    evidence_sentences: list,
    source_names: list = None,
    domain_names: list = None,
    nli_labels: list = None,
    timestamps: list = None,
    claim_temporal_signal: str = None,
    claim_date: str = None,
    claim_meta: dict = None,   # pre-computed by pipeline; skips a redundant LLM call
    request_id: str = "REQ-UNKNOWN",
) -> dict:
    """
    Entry point called by pipeline.py.
    Delegates to the multi-agent orchestrator.  When claim_meta is provided
    the orchestrator skips Phase A (analyze_claim) and uses it directly.
    """
    logger.info(
        f"[{request_id}] Delegating to multi-agent orchestrator. "
        f"claim='{claim[:60]}' evidence_count={len(evidence_sentences)}"
    )

    result = await run_verification_agents(
        claim=claim,
        evidence_sentences=evidence_sentences,
        source_names=source_names,
        domain_names=domain_names,
        nli_labels=nli_labels,
        timestamps=timestamps,
        claim_temporal_signal=claim_temporal_signal,
        claim_date=claim_date,
        claim_meta=claim_meta,
        request_id=request_id,
    )

    return result
