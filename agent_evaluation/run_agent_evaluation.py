import asyncio
import logging
import json
import os
import sys
from datetime import datetime

# Path setup
sys.path.append(os.getcwd())

from agent_evaluation.utils.pipeline_runner import run_pipeline
from agent_evaluation.evaluators.claim_extraction_evaluator import evaluate_claim_extraction
from agent_evaluation.evaluators.query_generator_evaluator import evaluate_query_generation
from agent_evaluation.evaluators.retrieval_evaluator import evaluate_retrieval
from agent_evaluation.evaluators.verifier_evaluator import evaluate_verdict
from agent_evaluation.evaluators.credibility_evaluator import evaluate_credibility
from agent_evaluation.metrics.ragas_metrics import compute_ragas_metrics
from agent_evaluation.metrics.deepeval_metrics import compute_deepeval_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentEvaluation")

async def main(limit=None):
    dataset_path = "agent_evaluation/dataset/trustlens_eval_dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    if limit:
        dataset = dataset[:limit]
        
    results = []
    logger.info(f"Starting evaluation on {len(dataset)} samples...")
    
    for i, sample in enumerate(dataset):
        logger.info(f"[{i+1}/{len(dataset)}] Sample ID: {sample.get('id')}")
        
        # 1. Run Pipeline Trace
        trace = await run_pipeline(sample["post"])
        if not trace:
            continue
            
        # 2. Compute Stage Metrics
        claims_metrics = evaluate_claim_extraction(trace["claims"], sample.get("claims", []))
        query_metrics = evaluate_query_generation(trace["queries"], trace["claims"][0])
        retrieval_metrics = evaluate_retrieval(trace["ranked_evidence"], sample.get("reference_evidence", []))
        verdict_metrics = evaluate_verdict(trace["verdict"], sample.get("ground_truth_verdict"))
        score_metrics = evaluate_credibility(trace["credibility_score"], sample.get("expected_score_range"))
        
        results.append({
            "sample_id": sample.get("id"),
            "trace": trace,
            "metrics": {
                **claims_metrics,
                **query_metrics,
                **retrieval_metrics,
                **verdict_metrics,
                **score_metrics
            }
        })
        
    # 3. Compute RAG/LLM Metrics
    ragas_data = [
        {
            "question": r["trace"]["claims"][0],
            "answer": r["trace"]["explanation"],
            "contexts": [e["text"] for e in r["trace"]["ranked_evidence"]],
            "ground_truth": r["trace"]["verdict"]
        } for r in results
    ]
    ragas_metrics = compute_ragas_metrics(ragas_data)
    
    deepeval_data = [
        {
            "input": r["trace"]["claims"][0],
            "context": [e["text"] for e in r["trace"]["ranked_evidence"]],
            "output": r["trace"]["explanation"]
        } for r in results
    ]
    deepeval_metrics = compute_deepeval_metrics(deepeval_data)
    
    # 4. Aggregate and Report
    success_results = [r for r in results]
    count = len(success_results)
    
    if count == 0:
        logger.error(
            "No pipeline traces succeeded. All samples failed (likely an API auth error). "
            "Check your AZURE_OPENAI_API_KEY in .env and ensure the subscription is active."
        )
        print("\n[ERROR] Zero successful pipeline traces. Cannot generate evaluation report.")
        print("Likely cause: Azure OpenAI API key is expired or invalid (HTTP 401).")
        print("Fix: Update AZURE_OPENAI_API_KEY in backend/.env then re-run the evaluation.")
        return
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(dataset),
        "successful_traces": count,
        "pipeline_metrics": {
            "claim_extraction_accuracy": sum(r["metrics"]["claim_extraction_accuracy"] for r in success_results) / count,
            "query_relevance_score": sum(r["metrics"]["query_relevance_score"] for r in success_results) / count,
            "retrieval_score": sum(r["metrics"]["retrieval_score"] for r in success_results) / count,
            "verdict_accuracy": sum(r["metrics"]["verdict_accuracy"] for r in success_results) / count,
            "credibility_score_accuracy": sum(r["metrics"]["credibility_score_accuracy"] for r in success_results) / count
        },
        "rag_metrics": ragas_metrics,
        "llm_metrics": deepeval_metrics,
        "detailed_results": results
    }
    
    # Save Report
    report_path = "agent_evaluation/reports/evaluation_report.json"
    os.makedirs("agent_evaluation/reports", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info("Evaluation Complete. Report saved to " + report_path)
    
    # Print Summary
    print("\nTrustLens Agent Evaluation Report")
    print("---------------------------------")
    print(f"Dataset Size: {count}")
    for k, v in report["pipeline_metrics"].items():
        print(f"{k.replace('_', ' ').title()}: {v:.2f}")
    for k, v in report["rag_metrics"].items():
        print(f"{k.replace('_', ' ').title()}: {v:.2f}")
    print(f"Hallucination Rate: {report['llm_metrics']['hallucination_rate']:.2f}")

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(main(limit))
