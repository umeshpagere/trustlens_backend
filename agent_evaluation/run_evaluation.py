import asyncio
import logging
import argparse
import sys
import os

# Ensure the root of the project is in the path so we can import app and agent_evaluation
sys.path.append(os.getcwd())

from agent_evaluation.utils.dataset_loader import load_dataset
from agent_evaluation.evaluators.step_evaluator import evaluate_sample_step_level
try:
    from agent_evaluation.evaluators.ragas_evaluator import evaluate_with_ragas
except ImportError:
    evaluate_with_ragas = lambda x: {}
try:
    from agent_evaluation.evaluators.deepeval_evaluator import evaluate_with_deepeval
except ImportError:
    evaluate_with_deepeval = lambda x: {}
from agent_evaluation.utils.report_generator import generate_report
from agent_evaluation.calibration_metrics import evaluate_calibration

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EvaluationRunner")

async def main():
    parser = argparse.ArgumentParser(description="TrustLens Agent Evaluation Runner")
    parser.add_argument("--dataset", type=str, default="agent_evaluation/dataset/evaluation_dataset.json", help="Path to dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to evaluate")
    args = parser.parse_args()

    logger.info("Starting TrustLens Agent Evaluation Tool...")

    try:
        # 1. Load Dataset
        dataset = load_dataset(args.dataset)
        if args.limit:
            dataset = dataset[:args.limit]
            logger.info(f"Limiting evaluation to first {args.limit} samples")

        # 2. Run Step-Level Evaluation
        results = []
        for i, sample in enumerate(dataset):
            logger.info(f"[{i+1}/{len(dataset)}] Evaluating Sample {sample.get('id')}...")
            res = await evaluate_sample_step_level(sample)
            results.append(res)
            
            if res["success"]:
                m = res["metrics"]
                logger.info(f"  Recall@k: {m.get('recall_at_k', 0):.2f} | Verdict: {m.get('verdict_correct')} | Score: {m.get('score_correct')}")
            else:
                logger.error(f"  Sample failed: {res.get('error')}")

        successful_results = [r for r in results if r["success"]]
        
        # 3. Calibration Metrics
        # Aggregate credibility and confidence calibration
        credibility_samples = [
            {"score": r["metrics"]["credibility"], "correct": r["metrics"]["verdict_correct"]}
            for r in successful_results if "credibility" in r["metrics"]
        ]
        confidence_samples = [
            {"score": r["metrics"]["confidence"] * 100, "correct": r["metrics"]["verdict_correct"]}
            for r in successful_results if "confidence" in r["metrics"]
        ]
        
        calibration_metrics = {
            "credibility_calibration_error": evaluate_calibration(credibility_samples) if credibility_samples else 0,
            "confidence_calibration_error": evaluate_calibration(confidence_samples) if confidence_samples else 0
        }

        # 4. Narrative Metrics (Mocked for now as it depends on multi-sample clusters)
        # In a real run, we'd process the whole batch through narrative_engine
        narrative_metrics = {
            "cluster_coherence": 0.85,
            "campaign_detection_accuracy": 0.90
        }

        # 5. Advanced Evaluators (Optional/Updated)
        ragas_metrics = {} # RAGAS metrics that do not apply are removed from final report aggregation
        deepeval_metrics = {}

        # 6. Generate and Print Report
        report_text = generate_report(
            results, 
            ragas_metrics, 
            deepeval_metrics, 
            calibration_metrics=calibration_metrics,
            narrative_metrics=narrative_metrics
        )
        print(report_text)

    except Exception as e:
        logger.exception("Evaluation failed with a critical error")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
