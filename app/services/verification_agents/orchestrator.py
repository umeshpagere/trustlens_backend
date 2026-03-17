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
import logging
from typing import Optional

from app.services.verification_agents.claim_agent     import analyze_claim
from app.services.verification_agents.evidence_agent  import evaluate_evidence
from app.services.verification_agents.source_agent    import analyze_sources
from app.services.verification_agents.temporal_agent  import check_temporal_consistency
from app.services.verification_agents.consensus_agent import synthesize_verdict
from app.services.evidence_pipeline.source_registry   import (
    TIER1_SOURCES, TIER2_SOURCES, TIER3_SOURCES, FACTCHECK_DOMAINS
)

logger = logging.getLogger(__name__)


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
) -> dict:
    """
    Orchestrate all five verification agents and return a result dict
    compatible with the existing downstream scoring contracts.

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
        return _fallback

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
    # Phase A — sequential: Claim Analysis (output used by Phase B agents)
    # -------------------------------------------------------------------------
    logger.info("[Orchestrator] Phase A — Claim Analysis")
    claim_analysis = await analyze_claim(claim)

    # Fill temporal signals from agent output if not already provided
    resolved_temporal_signal = claim_temporal_signal or claim_analysis.get("temporal_signal")
    resolved_explicit_date   = claim_date            or claim_analysis.get("explicit_date")

    # -------------------------------------------------------------------------
    # Phase B — parallel: Evidence + Source + Temporal agents
    # -------------------------------------------------------------------------
    logger.info("[Orchestrator] Phase B — Evidence / Source / Temporal (parallel)")
    evidence_coro = evaluate_evidence(claim, evidence_dicts)
    source_coro   = analyze_sources(evidence_dicts)
    temporal_coro = check_temporal_consistency(
        claim,
        evidence_dicts,
        temporal_signal=resolved_temporal_signal,
        explicit_date=resolved_explicit_date,
    )

    evidence_analysis, source_analysis, temporal_analysis = await asyncio.gather(
        evidence_coro, source_coro, temporal_coro,
        return_exceptions=False,
    )

    # -------------------------------------------------------------------------
    # Phase C — sequential: Consensus Synthesizer
    # -------------------------------------------------------------------------
    logger.info("[Orchestrator] Phase C — Consensus Synthesis")
    consensus = await synthesize_verdict(
        claim,
        claim_analysis,
        evidence_analysis,
        source_analysis,
        temporal_analysis,
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

    return {
        "verdict":                 consensus.get("verdict", "UNVERIFIED"),
        "credibility_score":       consensus.get("credibility_score", 50),
        "confidence":              consensus.get("confidence", 0.3),
        "explanation":             consensus.get("reasoning", ""),
        "reasoning_steps":         reasoning_steps,
        "trusted_sources":         trusted,
        "low_credibility_sources": low_cred,
        "source_agreement":        agreement,
        "agent_outputs": {
            "claim_analysis":    claim_analysis,
            "evidence_analysis": evidence_analysis,
            "source_analysis":   source_analysis,
            "temporal_analysis": temporal_analysis,
        },
    }
