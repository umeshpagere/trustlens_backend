"""
Multi-Agent Orchestrator — Part-11

Coordinates the five verification agents:
  1. Claim Analysis Agent     (sequential — its output feeds agents 2–4)
  2. Evidence Validation Agent (parallel with 3 & 4)
  3. Source Credibility Agent  (parallel)
  4. Temporal Consistency Agent (parallel)
  5. Consensus Synthesizer     (sequential — consumes all four outputs)

Architecture (timing):
  Phase A: analyze_claim()                          → claim_analysis
  Phase B: asyncio.gather(                          → evidence/source/temporal in parallel
               evaluate_evidence(),
               analyze_sources(),
               check_temporal_consistency()
           )
  Phase C: synthesize_verdict()                     → final verdict

The return dict from run_verification_agents() is compatible with the
existing credibility_engine and confidence_service contracts.
"""

import asyncio
import hashlib
import logging
from typing import Optional

from app.services.verification_agents.claim_agent     import analyze_claim
from app.services.verification_agents.evidence_agent  import evaluate_evidence
from app.services.verification_agents.source_agent    import analyze_sources
from app.services.verification_agents.temporal_agent  import check_temporal_consistency
from app.services.verification_agents.consensus_agent import synthesize_verdict
from app.services.verification_agents._utils          import AsyncCache
from app.services.evidence_pipeline.source_registry   import (
    TIER1_SOURCES, TIER2_SOURCES, TIER3_SOURCES, FACTCHECK_DOMAINS
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Short-TTL result cache — avoids redundant LLM calls for identical requests
# that arrive within 5 minutes (e.g. duplicate submissions, retries).
# ---------------------------------------------------------------------------
_result_cache: AsyncCache = AsyncCache(ttl_seconds=300)


def _build_evidence_dicts(
    evidence_sentences: list,
    source_names: list,
    domain_names: list,
    nli_labels: list,
    timestamps: list,
) -> list:
    """
    Convert flat parallel lists into a list of evidence dicts for agents.
    All lists may be shorter than evidence_sentences; safe defaults applied.
    """
    result = []
    for i, text in enumerate(evidence_sentences):
        result.append({
            "id":           i + 1,           # 1-indexed for LLM prompts
            "text":         text,
            "source":       source_names[i]  if source_names  and i < len(source_names)  else f"Source {i+1}",
            "domain":       domain_names[i]  if domain_names  and i < len(domain_names)  else "unknown",
            "nli_hint":     nli_labels[i]    if nli_labels    and i < len(nli_labels)    else "unknown",
            "published_at": timestamps[i]    if timestamps    and i < len(timestamps)    else None,
        })
    return result


def _compute_source_metrics(
    evidence_analysis: dict,
    domain_names: list,
) -> tuple[list, list, float]:
    """
    Derive trusted_sources, low_credibility_sources, and source_agreement
    from the evidence agent's evaluations and deterministic domain registry.
    Keeps Part-9 source metrics alive in Part-11 output.
    """
    evaluations = evidence_analysis.get("evaluations", [])

    trusted       = set()
    low_cred      = set()
    supported_dom = set()
    contradicted_dom = set()

    for ev in evaluations:
        idx = ev.get("evidence_id", 0) - 1      # convert 1-indexed → 0-indexed
        relation = ev.get("relation", "NEUTRAL")
        domain = domain_names[idx] if 0 <= idx < len(domain_names) else "unknown"

        if domain in TIER1_SOURCES or domain in TIER2_SOURCES or domain in FACTCHECK_DOMAINS:
            trusted.add(domain)
        elif domain in TIER3_SOURCES:
            low_cred.add(domain)

        if relation == "SUPPORT":
            supported_dom.add(domain)
        elif relation == "CONTRADICT":
            contradicted_dom.add(domain)

    total = len(supported_dom | contradicted_dom)
    agreement = len(supported_dom) / total if total else 0.0

    return sorted(trusted), sorted(low_cred), round(agreement, 3)


async def run_verification_agents(
    claim: str,
    evidence_sentences: list,
    source_names: list = None,
    domain_names: list = None,
    nli_labels: list = None,
    timestamps: list = None,
    claim_temporal_signal: str = None,
    claim_date: str = None,
    claim_meta: dict = None,   # pre-computed by pipeline; skips redundant Phase A LLM call
    include_agent_traces: bool = False, # flag to include full agent outputs
    request_id: str = "REQ-UNKNOWN",
) -> dict:
    """
    Orchestrate multi-agent verification for a single claim.

    When claim_meta is provided, Phase A (analyze_claim LLM call) is skipped
    and the pre-computed result is used directly, saving one Azure OpenAI call
    per claim.

    Output schema (backward-compatible with monolithic verify_claim_credibility):
    {
        "verdict":                  "SUPPORTED | CONTRADICTED | UNVERIFIED",
        "credibility_score":        int 0-100,
        "confidence":               float 0.0-1.0,
        "explanation":              str,
        "reasoning_steps":          list,        # from evidence agent evaluations
        "trusted_sources":          list[str],
        "low_credibility_sources":  list[str],
        "source_agreement":         float,
        "agent_outputs": {                        # full agent outputs for debugging
            "claim_analysis":    dict,
            "evidence_analysis": dict,
            "source_analysis":   dict,
            "temporal_analysis": dict,
        }
    }
    """
    logger.info(f"[{request_id}] Orchestrating verification for claim: {claim[:50]}...")

    _fallback = {
        "verdict":                 "UNVERIFIED",
        "credibility_score":       50,
        "confidence":              0.3,
        "explanation":             "Multi-agent verification failed. No evidence available.",
        "reasoning_steps":         [],
        "trusted_sources":         [],
        "low_credibility_sources": [],
        "source_agreement":        0.0,
        "agent_outputs":           {},
    }

    if not evidence_sentences:
        logger.warning(f"[{request_id}] No evidence provided — returning fallback.")
        return _fallback

    # ---- Cache lookup -------------------------------------------------------
    # Build a deterministic key from the claim text and the sorted evidence set
    # so that order-shuffled duplicates still hit the cache.
    _evidence_fingerprint = "|".join(sorted(str(s) for s in evidence_sentences))
    _cache_key = hashlib.sha256(
        f"{claim.strip()}|||{_evidence_fingerprint}".encode("utf-8")
    ).hexdigest()
    _cached = _result_cache.get(_cache_key)
    if _cached is not None:
        logger.info(f"[{request_id}] Cache HIT — returning cached orchestrator result")
        return _cached

    # Normalise nullable lists
    source_names = source_names or []
    domain_names = domain_names or []
    nli_labels   = nli_labels   or []
    timestamps   = timestamps   or []

    # Build structured evidence dicts consumed by agents 2, 3, 4
    evidence_dicts = _build_evidence_dicts(
        evidence_sentences, source_names, domain_names, nli_labels, timestamps
    )

    # -------------------------------------------------------------------------
    # Phase A — sequential: Claim Analysis (reuse pre-computed or call LLM)
    # -------------------------------------------------------------------------
    if claim_meta is not None:
        logger.info(f"[{request_id}] Phase A — Skipped (claim_meta provided by pipeline)")
        claim_analysis = claim_meta
    else:
        logger.info(f"[{request_id}] Phase A — Claim Analysis")
        claim_analysis = await analyze_claim(claim, request_id=request_id)

    # Fill temporal signals from agent output if not already provided
    resolved_temporal_signal = claim_temporal_signal or claim_analysis.get("temporal_signal")
    resolved_explicit_date   = claim_date            or claim_analysis.get("explicit_date")

    # -------------------------------------------------------------------------
    # Phase B — parallel: Evidence + Source + Temporal agents
    # -------------------------------------------------------------------------
    logger.info("[Orchestrator] Phase B — Evidence / Source / Temporal (parallel)")
    # Wrap each coroutine with a 20-second timeout so a slow Azure call
    # never blocks the entire pipeline.
    tasks = [
        asyncio.wait_for(evaluate_evidence(claim, claim_analysis, evidence_dicts, request_id=request_id), timeout=20.0),
        asyncio.wait_for(analyze_sources(claim, claim_analysis, evidence_dicts, request_id=request_id), timeout=20.0),
        asyncio.wait_for(
            check_temporal_consistency(
                claim,
                claim_analysis,
                evidence_dicts,
                temporal_signal=resolved_temporal_signal,
                explicit_date=resolved_explicit_date,
                request_id=request_id,
            ),
            timeout=20.0,
        ),
    ]

    # return_exceptions=True: if any one agent fails the others still complete.
    phase_b_results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    _ev_fallback  = {"evaluations": []}
    _src_fallback = {"source_evaluations": [], "credibility_index": 0.5}
    _tmp_fallback = {"temporal_assessment": [], "temporal_consistency_score": 0.5}

    if isinstance(phase_b_results[0], Exception):
        logger.warning("[Orchestrator] EvidenceAgent failed: %s", phase_b_results[0])
        evidence_analysis = _ev_fallback
    else:
        evidence_analysis = phase_b_results[0]

    if isinstance(phase_b_results[1], Exception):
        logger.warning("[Orchestrator] SourceAgent failed: %s", phase_b_results[1])
        source_analysis = _src_fallback
    else:
        source_analysis = phase_b_results[1]

    if isinstance(phase_b_results[2], Exception):
        logger.warning("[Orchestrator] TemporalAgent failed: %s", phase_b_results[2])
        temporal_analysis = _tmp_fallback
    else:
        temporal_analysis = phase_b_results[2]

    # -------------------------------------------------------------------------
    # Phase C — sequential: Consensus Synthesizer
    # -------------------------------------------------------------------------
    logger.info(f"[{request_id}] Phase C — Consensus Synthesis")
    consensus = await synthesize_verdict(
        claim,
        claim_analysis,
        evidence_analysis,
        source_analysis,
        temporal_analysis,
        request_id=request_id,
    )

    # -------------------------------------------------------------------------
    # Compute Part-9 source metrics from evidence evaluations
    # -------------------------------------------------------------------------
    trusted, low_cred, agreement = _compute_source_metrics(
        evidence_analysis, domain_names
    )

    # -------------------------------------------------------------------------
    # Convert evidence agent evaluations → reasoning_steps schema
    # (compatible with Part-10 Pydantic schema downstream)
    # -------------------------------------------------------------------------
    reasoning_steps = [
        {
            "evidence_id": ev.get("evidence_id"),
            "relation":    ev.get("relation", "NEUTRAL"),
            "reason":      ev.get("reason", ""),
        }
        for ev in evidence_analysis.get("evaluations", [])
    ]

    result = {
        "verdict":                 consensus.get("verdict", "UNVERIFIED"),
        "credibility_score":       consensus.get("credibility_score", 50),
        "confidence":              consensus.get("confidence", 0.3),
        "explanation":             consensus.get("reasoning", ""),
        "reasoning_steps":         reasoning_steps,
        "trusted_sources":         trusted,
        "low_credibility_sources": low_cred,
        "source_agreement":        agreement,
    }

    if include_agent_traces:
        result["agent_outputs"] = {
            "claim_analysis":    claim_analysis,
            "evidence_analysis": evidence_analysis,
            "source_analysis":   source_analysis,
            "temporal_analysis": temporal_analysis,
        }

    # ---- Populate cache (only cache non-trace results to keep entries small) ---
    if not include_agent_traces:
        _result_cache.set(_cache_key, result)
        logger.debug(f"[{request_id}] Cache SET for key {_cache_key[:12]}...")

    return result
