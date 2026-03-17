"""
Adaptive Retrieval Controller — Part-12

Orchestrates:
  1. plan_retrieval()   → planner decides sources + query count
  2. expand_queries()   → generates expanded query list
  3. aggregate_evidence(queries, sources=...)  → first retrieval pass
  4. compute_coverage() → check if evidence is sufficient
  5. If insufficient and loops remaining → second pass with fallback queries
  6. Return combined deduplicated evidence + retrieval metadata

Hard limits:
  MAX_QUERIES_TOTAL = 8    (enforced by planner)
  MAX_RETRIEVAL_LOOPS = 2  (prevents runaway API usage)
  TIME_LIMIT_SECONDS = 45  (wall-clock guard for the whole retrieval phase)
"""

import logging
import time
import hashlib

from app.services.retrieval_planner.planner_agent  import plan_retrieval
from app.services.retrieval_planner.query_expander import expand_queries
from app.services.retrieval_planner.coverage_metrics import (
    compute_coverage, is_coverage_sufficient, COVERAGE_SUFFICIENT
)
from app.services.evidence_pipeline.evidence_aggregator import aggregate_evidence

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Raised when retrieval completes but returns zero documents."""
    pass

MAX_RETRIEVAL_LOOPS = 2
TIME_LIMIT_SECONDS  = 45


def _deduplicate(docs: list) -> list:
    """Remove duplicate documents by URL (fallback: deterministic hash of text)."""
    seen = set()
    unique = []
    for doc in docs:
        url = doc.get("url", "")
        text = doc.get("text", "") or ""
        key = url if url else hashlib.md5(text.encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _fallback_queries(claim: str, entities: list) -> list:
    """Simple additional queries used in the second retrieval loop."""
    extras = [f"{claim} evidence", f"{claim} report"]
    for e in entities[:2]:
        extras.append(f"{e} latest news")
    return extras[:4]


async def controlled_retrieval(
    claim: str,
    entities: list,
    claim_meta: dict,
) -> tuple[list, dict]:
    """
    Adaptive retrieval controller.

    Returns:
        (evidence_docs: list, retrieval_meta: dict)

    retrieval_meta keys:
        sources_used      list[str]
        queries_used      list[str]
        plan              dict
        final_coverage    float
        retrieval_loops   int
        retrieved_docs    int
    """
    t_start = time.monotonic()

    # -------------------------------------------------------------------------
    # Phase 1 — Plan
    # -------------------------------------------------------------------------
    plan = await plan_retrieval(claim, claim_meta, entities)
    sources   = plan["sources"]
    max_q     = plan["max_queries"]
    temp_sig  = claim_meta.get("temporal_signal", "UNKNOWN")

    logger.info(f"[Controller] Plan: sources={sources} max_queries={max_q} depth={plan['retrieval_depth']}")

    # -------------------------------------------------------------------------
    # Phase 2 — Query Expansion
    # -------------------------------------------------------------------------
    queries = await expand_queries(claim, entities, plan, temporal_signal=temp_sig)
    logger.info(f"[Controller] Expanded queries ({len(queries)}): {queries}")

    all_evidence: list = []
    loops_done  = 0
    final_coverage = 0.0
    queries_used = list(queries)

    # -------------------------------------------------------------------------
    # Phase 3 — Retrieval loop (max MAX_RETRIEVAL_LOOPS passes)
    # -------------------------------------------------------------------------
    for loop in range(MAX_RETRIEVAL_LOOPS):
        # Time guard
        elapsed = time.monotonic() - t_start
        if elapsed > TIME_LIMIT_SECONDS:
            logger.warning(f"[Controller] Time limit {TIME_LIMIT_SECONDS}s reached, stopping retrieval")
            break

        logger.info(f"[Controller] Retrieval loop {loop + 1}/{MAX_RETRIEVAL_LOOPS} "
                    f"with {len(queries)} queries, sources={sources}")

        # Call aggregate_evidence with source routing (Step 8)
        logger.info(f"[Controller] Queries generated: {queries}")
        batch = await aggregate_evidence(queries, sources=sources)
        all_evidence.extend(batch)
        all_evidence = _deduplicate(all_evidence)
        loops_done   = loop + 1

        final_coverage = compute_coverage(all_evidence)
        sufficient, _ = is_coverage_sufficient(all_evidence)

        logger.info(
            f"[Controller] After loop {loop + 1}: "
            f"docs={len(all_evidence)} coverage={final_coverage:.2f} "
            f"sufficient={sufficient}"
        )

        if sufficient:
            logger.info("[Controller] Coverage sufficient — stopping retrieval")
            break

        if loop + 1 < MAX_RETRIEVAL_LOOPS:
            # Generate fallback queries for the next loop
            queries = _fallback_queries(claim, entities)
            queries_used.extend(queries)
            logger.info(f"[Controller] Coverage weak — triggering loop {loop + 2} with fallback queries: {queries}")

    if not all_evidence:
        logger.error("[Controller] Retrieval completed with zero documents.")
        raise RetrievalError("No documents retrieved for claim")

    return all_evidence, {
        "sources_used":   sources,
        "queries_used":   queries_used,
        "plan":           plan,
        "final_coverage": final_coverage,
        "retrieval_loops": loops_done,
        "retrieved_docs": len(all_evidence),
    }
