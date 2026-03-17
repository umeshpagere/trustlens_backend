import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_report(results, ragas_metrics, deepeval_metrics, calibration_metrics=None, narrative_metrics=None):
    """
    Generates a final evaluation report from a list of sample results with multiple layers.
    """
    if not results:
        return "No evaluation results to report."

    total_samples = len(results)
    successful_samples = [r for r in results if r["success"]]
    success_count = len(successful_samples)

    if success_count == 0:
        return "All evaluations failed."

    # 1. Retrieval Metrics Aggregation
    avg_recall = sum(r["metrics"].get("recall_at_k", 0) for r in successful_samples) / success_count
    avg_precision = sum(r["metrics"].get("precision_at_k", 0) for r in successful_samples) / success_count
    avg_diversity = sum(r["metrics"].get("source_diversity", 0) for r in successful_samples) / success_count
    avg_efficiency = sum(r["metrics"].get("planner_efficiency", 0) for r in successful_samples) / success_count

    # 2. Agent Metrics Aggregation
    avg_claim_acc = sum(r["metrics"].get("claim_structure_accuracy", 0) for r in successful_samples) / success_count
    # Note: source_agent_accuracy, etc. would be aggregated here if collected per sample

    # 3. System Metrics Aggregation
    verdict_accuracy = sum(1 for r in successful_samples if r["metrics"].get("verdict_correct")) / success_count
    avg_stability = sum(r["metrics"].get("stability_score", 0) for r in successful_samples) / success_count

    report_text = f"""
==============================
TrustLens Evaluation Framework Upgraded
==============================
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset size: {total_samples}
Successful evaluations: {success_count}

1. Retrieval Metrics
------------------
Recall@k: {avg_recall:.2f}
Precision@k: {avg_precision:.2f}
Source Diversity: {avg_diversity:.2f}
Planner Efficiency: {avg_efficiency:.2f}

2. Agent Metrics
------------------
Claim Agent Accuracy: {avg_claim_acc:.2f}
Verdict Stability: {avg_stability:.2f}

3. System Metrics
------------------
Verdict Accuracy: {verdict_accuracy:.2f}
Credibility Calibration Error: {calibration_metrics.get('credibility_calibration_error', 0) if calibration_metrics else 0:.2f}
Confidence Calibration Error: {calibration_metrics.get('confidence_calibration_error', 0) if calibration_metrics else 0:.2f}

4. Narrative Metrics
------------------
Cluster Coherence: {narrative_metrics.get('cluster_coherence', 0) if narrative_metrics else 0:.2f}
Campaign Detection Accuracy: {narrative_metrics.get('campaign_detection_accuracy', 0) if narrative_metrics else 0:.2f}

==============================
"""
    
    import numpy as np
    import math

    def _to_json_serializable(obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json_serializable(i) for i in obj]
        return str(obj)

    # Save machine-readable report
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": total_samples,
        "retrieval_metrics": {
            "recall_at_k": float(avg_recall),
            "precision_at_k": float(avg_precision),
            "source_diversity": float(avg_diversity),
            "planner_efficiency": float(avg_efficiency)
        },
        "agent_metrics": {
            "claim_agent_accuracy": float(avg_claim_acc),
            "verdict_stability": float(avg_stability)
        },
        "system_metrics": {
            "verdict_accuracy": float(verdict_accuracy),
            "credibility_calibration_error": float(calibration_metrics.get('credibility_calibration_error', 0) if calibration_metrics else 0),
            "confidence_calibration_error": float(calibration_metrics.get('confidence_calibration_error', 0) if calibration_metrics else 0)
        },
        "narrative_metrics": _to_json_serializable(narrative_metrics or {}),
        "detailed_results": _to_json_serializable(results)
    }

    report_path = "agent_evaluation/evaluation_report.json"
    try:
        # Create directory if doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")

    return report_text
