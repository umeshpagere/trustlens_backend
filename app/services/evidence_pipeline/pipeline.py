import ast
import logging
import asyncio
from .entity_extraction import extract_entities
from .query_generator import generate_queries
from .evidence_aggregator import aggregate_evidence
from .semantic_ranker import rank_evidence
from .evidence_alignment import align_evidence, align_evidence_fast
from .nli_verifier import check_contradiction
from app.services.verification_engine import verify_claim_credibility
from .claim_strength_filter import filter_claims
from app.services.retrieval_planner.retrieval_controller import (
    controlled_retrieval,
    RetrievalError,
)
from app.services.verification_agents.claim_agent import analyze_claim

logger = logging.getLogger(__name__)

# Part 7: Concurrency limit for Azure OpenAI
verification_semaphore = asyncio.Semaphore(5)

async def _process_single_claim(item, claim_idx, total_claims, request_id="REQ-UNKNOWN"):
    """Internal helper to process a single claim asynchronously."""
    claim = item.get("text", "")
    # Guard: unwrap Python dict-repr strings e.g. "{'text': '...', 'temporal_signal': None}"
    if isinstance(claim, str) and claim.startswith("{") and "'text'" in claim:
        try:
            parsed = ast.literal_eval(claim)
            if isinstance(parsed, dict):
                claim = parsed.get("text", claim)
        except (ValueError, SyntaxError):
            pass
    source = item.get("source", "unknown")
    # Part-10: extract temporal metadata from the structured claim object
    temporal_signal = item.get("temporal_signal")
    explicit_date   = item.get("explicit_date")
    
    logger.info(f"[{request_id}] [{claim_idx}/{total_claims}] Processing claim: {claim[:50]}...")
    
    try:
        # 1. Entity extraction
        entities = item.get("entities") or extract_entities(claim)

        # 2. Claim Analysis (reuses Part-11 agent; provides event_type + expected_evidence_types)
        claim_meta = await analyze_claim(claim, request_id=request_id)
        # Merge temporal info from claim struct (pipeline-set) over agent output
        if temporal_signal:
            claim_meta["temporal_signal"] = temporal_signal
        if explicit_date:
            claim_meta["explicit_date"] = explicit_date

        # 3. Adaptive Retrieval (planner + query expansion + coverage loop)
        #    Strictly: Claim → Queries → Retrieval → Candidate Pool
        try:
            evidence_docs, retrieval_meta = await controlled_retrieval(
                claim=claim,
                entities=entities,
                claim_meta=claim_meta,
            )
        except RetrievalError as exc:
            logger.error(f"[{claim_idx}/{total_claims}] RetrievalError: {exc}")
            return {
                "claim": claim,
                "source": source,
                "entities": entities,
                "queries": [],
                "verification": {
                    "verdict": "UNVERIFIED",
                    "credibility_score": 50,
                    "explanation": "No documents could be retrieved for this claim.",
                },
                "evidence": [],
                "stats": {"retrieved_docs": 0, "aligned_sentences": 0},
            }

        logger.info(
            f"[{claim_idx}/{total_claims}] Retrieved documents: {len(evidence_docs)} "
            f"| coverage={retrieval_meta.get('final_coverage', 0):.2f} "
            f"| loops={retrieval_meta.get('retrieval_loops', 1)}"
        )

        # Keep the raw candidate document pool for evaluation/debugging.
        # (Alignment will convert docs -> sentences and apply NLI filtering.)
        raw_docs = list(evidence_docs)

        # 4. Fast Alignment — semantic similarity only (no NLI bottleneck)
        #    Takes ~0.5s instead of 30-90s. LLM agents will do SUPPORT/CONTRADICT classification.
        aligned = await asyncio.to_thread(
            align_evidence_fast, claim, evidence_docs, top_n=5
        )
        
        logger.info(f"[{claim_idx}/{total_claims}] Evidence aligned: {len(aligned)}")

        # 5. Verify claim using multi-agent engine with Semaphore
        sentences_for_verification = [s["text"] for s in aligned]
        source_names = [s.get("source", "Unknown") for s in aligned]
        domain_names = [s.get("domain", "unknown") for s in aligned]
        nli_labels   = [s.get("nli_label", "unknown") for s in aligned]
        # Fix-2: forward per-sentence publish timestamps to the temporal agent
        timestamps   = [s.get("timestamp") or s.get("published_at") for s in aligned]
        
        async with verification_semaphore:
            verification = await verify_claim_credibility(
                claim,
                sentences_for_verification,
                source_names=source_names,
                domain_names=domain_names,
                nli_labels=nli_labels,
                timestamps=timestamps,
                claim_temporal_signal=temporal_signal,
                claim_date=explicit_date,
                claim_meta=claim_meta,   # re-use already-computed claim analysis
                request_id=request_id,
            )
        
        logger.info(f"[{claim_idx}/{total_claims}] Verification result: {verification.get('verdict')}")

        return {
            "claim": claim,
            "source": source,
            "entities": entities,
            "queries": retrieval_meta.get("queries_used", []),
            "verification": verification,
            # Backward-compatible key: aligned evidence sentences
            "evidence": aligned,
            # Explicit stage artifacts for evaluators/debuggers
            "raw_docs": raw_docs,
            "aligned_sentences": aligned,
            "retrieval_meta": retrieval_meta,
            "stats": {
                "retrieved_docs":    len(evidence_docs),
                "aligned_sentences": len(aligned),
                "coverage_score":    retrieval_meta.get("final_coverage", 0),
                "retrieval_loops":   retrieval_meta.get("retrieval_loops", 1),
            }
        }
    except Exception as e:
        logger.error(f"Error processing claim {claim_idx}: {e}")
        return {
            "claim": claim,
            "source": source,
            "verification": {"verdict": "ERROR", "explanation": str(e), "credibility_score": 50},
            "evidence": [],
            "stats": {"retrieved_docs": 0, "aligned_sentences": 0}
        }

async def run_evidence_pipeline(claims_data, request_id="REQ-UNKNOWN"):
    """
    Run the evidence verification pipeline on a list of claims (Asynchronous Parallel).
    """
    if not claims_data:
        logger.warning("No claims provided to evidence pipeline.")
        return []

    # 1. PRE-FILTERING: Remove weak, metadata or question claims
    raw_claims = [c.get("text", "") for c in claims_data if c.get("text")]
    filtered_texts = filter_claims(raw_claims)
    
    # Reconstruct claims_data with only the filtered results.
    # Build an O(1) index to avoid the previous O(N²) linear scan.
    claims_index = {item.get("text", ""): item for item in claims_data if item.get("text")}
    filtered_claims_data = [
        claims_index[text] for text in filtered_texts if text in claims_index
    ]

    total_verifiable = len(filtered_claims_data)
    logger.info(f"Pipeline: {len(raw_claims)} inputs -> {total_verifiable} verifiable claims.")
    
    if total_verifiable == 0:
        return []

    # 2. Parallel Execution using asyncio.gather
    tasks = [
        _process_single_claim(item, i + 1, total_verifiable, request_id=request_id) 
        for i, item in enumerate(filtered_claims_data)
    ]
    
    results = await asyncio.gather(*tasks)

    logger.info(f"Evidence pipeline completed for {len(results)} claims.")
    return results

def compute_evidence_score(results: list) -> float:
    """
    Calculate evidence score based on verified claims.
    SUPPORTED → 1.0
    UNVERIFIED → 0.5
    CONTRADICTED → 0
    """
    if not results:
        return 50.0  # Neutral baseline

    score_map = {
        "SUPPORTED": 1.0,
        "UNVERIFIED": 0.5,
        "CONTRADICTED": 0.0
    }

    total = 0
    valid_results = [r for r in results if "verification" in r]
    
    if not valid_results:
        return 50.0

    for r in valid_results:
        verdict = r["verification"].get("verdict", "UNVERIFIED")
        total += score_map.get(verdict, 0.5)

    return (total / len(valid_results)) * 100
