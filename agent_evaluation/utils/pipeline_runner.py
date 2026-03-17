import logging
import asyncio
from app.services.llm_analysis import analyze_text_with_llm
from app.services.evidence_pipeline.pipeline import run_evidence_pipeline

logger = logging.getLogger(__name__)

async def run_pipeline(post_text):
    """
    Executes the full TrustLens verification pipeline and captures a detailed trace.
    
    Trace mapping:
    - claims: extracted via analyze_text_with_llm
    - queries: expanded queries from retrieval planner
    - retrieved_docs: raw documents list
    - ranked_evidence: documents after ranking/alignment
    - verdict: final consensus verdict
    - explanation: reasoning from consensus agent
    - credibility_score: final confidence/credibility
    """
    logger.info(f"Running pipeline trace for: {post_text[:50]}...")
    
    # 1. Claim Extraction
    llm_result = await analyze_text_with_llm(post_text)
    claims_struct = llm_result.get("claims", [])
    claims = [c.get("text") if isinstance(c, dict) else c for c in claims_struct]
    
    if not claims:
        return None
        
    # 2. Evidence Pipeline (Multiple agents inside)
    claims_data = [{"text": c, "source": "llm_extraction"} for c in claims]
    pipeline_results = await run_evidence_pipeline(claims_data)
    
    # For evaluation, we typically focus on the primary/first claim
    primary_res = pipeline_results[0] if pipeline_results else {}
    verification = primary_res.get("verification", {})
    retrieval_meta = primary_res.get("retrieval_meta", {})
    
    trace = {
        "post": post_text,
        "claims": claims,
        "queries": retrieval_meta.get("queries_used", []),
        "retrieved_docs": [d.get("text", "") for d in primary_res.get("raw_docs", [])],
        "ranked_evidence": primary_res.get("aligned_sentences", primary_res.get("evidence", [])),
        "verdict": verification.get("verdict"),
        "explanation": verification.get("explanation"),
        "credibility_score": verification.get("credibility_score")
    }
    
    return trace
