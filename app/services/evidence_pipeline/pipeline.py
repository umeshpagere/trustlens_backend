import logging
import asyncio
from .entity_extraction import extract_entities
from .query_generator import generate_queries
from .evidence_aggregator import aggregate_evidence
from .semantic_ranker import rank_evidence
from .evidence_alignment import align_evidence
from .nli_verifier import check_contradiction
from .evidence_verifier import verify_claim_with_evidence
from .claim_strength_filter import filter_claims

logger = logging.getLogger(__name__)

# Part 7: Concurrency limit for Azure OpenAI
verification_semaphore = asyncio.Semaphore(5)

async def _process_single_claim(item, claim_idx, total_claims):
    """Internal helper to process a single claim asynchronously."""
    claim = item.get("text", "")
    source = item.get("source", "unknown")
    
    logger.info(f"[{claim_idx}/{total_claims}] Processing claim: {claim}")
    
    try:
        # 1. Extraction & Query Gen (Sync, can be wrapped if cpu-heavy)
        entities = extract_entities(claim)
        queries = generate_queries(claim, entities)
        
        # 2. Parallel Evidence Retrieval (Internal ThreadPool in aggregate_evidence)
        evidence_docs = aggregate_evidence(queries)
        logger.info(f"[{claim_idx}/{total_claims}] Evidence retrieved: {len(evidence_docs)}")
        
        if not evidence_docs:
             return {
                 "claim": claim,
                 "source": source,
                 "entities": entities,
                 "queries": queries,
                 "verification": {
                     "verdict": "UNVERIFIED",
                     "confidence": 0.0,
                     "reasoning": "No reliable evidence retrieved."
                 },
                 "evidence": []
             }
             
        # 3. Alignment (Sync)
        aligned = align_evidence(claim, evidence_docs, use_nli=True)
        logger.info(f"[{claim_idx}/{total_claims}] Evidence aligned: {len(aligned)}")

        # 4. Verify claim using LLM with Semaphore (Async)
        sentences_for_verification = [s["text"] for s in aligned]
        source_names = [s.get("source", "Unknown") for s in aligned]
        
        async with verification_semaphore:
            verification = await verify_claim_with_evidence(claim, sentences_for_verification, source_names=source_names)
        
        logger.info(f"[{claim_idx}/{total_claims}] Verification result: {verification.get('verdict')}")

        return {
            "claim": claim,
            "source": source,
            "entities": entities,
            "queries": queries,
            "verification": verification,
            "evidence": aligned
        }
    except Exception as e:
        logger.error(f"Error processing claim {claim_idx}: {e}")
        return {
            "claim": claim,
            "source": source,
            "verification": {"verdict": "ERROR", "reasoning": str(e)},
            "evidence": []
        }

async def run_evidence_pipeline(claims_data):
    """
    Run the evidence verification pipeline on a list of claims (Asynchronous Parallel).
    """
    if not claims_data:
        logger.warning("No claims provided to evidence pipeline.")
        return []

    # 1. PRE-FILTERING: Remove weak, metadata or question claims
    raw_claims = [c.get("text", "") for c in claims_data if c.get("text")]
    filtered_texts = filter_claims(raw_claims)
    
    # Reconstruct claims_data with only the filtered results
    filtered_claims_data = []
    for text in filtered_texts:
        original_item = next((item for item in claims_data if item.get("text") == text), None)
        if original_item:
            filtered_claims_data.append(original_item)

    total_verifiable = len(filtered_claims_data)
    logger.info(f"Pipeline: {len(raw_claims)} inputs -> {total_verifiable} verifiable claims.")
    
    if total_verifiable == 0:
        return []

    # 2. Parallel Execution using asyncio.gather
    tasks = [
        _process_single_claim(item, i + 1, total_verifiable) 
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
