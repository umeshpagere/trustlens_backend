import logging
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from app.config.settings import Config

logger = logging.getLogger(__name__)

def evaluate_with_deepeval(samples):
    """
    Evaluates samples for hallucinations using DeepEval.
    
    Args:
        samples (list[dict]): List of samples, each with:
            - input: str (the raw post)
            - actual_output: str (the agent's explanation)
            - retrieval_context: list[str] (retrieved evidence)
            
    Returns:
        dict: Average hallucination score.
    """
    if not samples:
        return {"hallucination_rate": 0.0}

    logger.info(f"Running DeepEval hallucination check on {len(samples)} samples")

    scores = []
    
    try:
        from deepeval.models import AzureOpenAIModel
        # Configure DeepEval to use Azure OpenAI
        azure_model = AzureOpenAIModel(
            model="gpt-4o-mini",
            deployment_name=Config.AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )
        metric = HallucinationMetric(threshold=0.5, model=azure_model)
    except Exception as e:
        logger.error(f"Failed to initialize DeepEval Azure model: {e}")
        return {"hallucination_rate": 0.0, "error": str(e)}

    for i, sample in enumerate(samples):
        try:
            test_case = LLMTestCase(
                input=sample.get("input", ""),
                actual_output=sample.get("actual_output", ""),
                retrieval_context=sample.get("retrieval_context", []),
                context=sample.get("retrieval_context", [])
            )
            metric.measure(test_case)
            scores.append(metric.score)
            logger.info(f"Sample {i+1} Hallucination Score: {metric.score}")
        except Exception as e:
            logger.error(f"DeepEval failed for sample {i}: {e}")
            scores.append(1.0) # Assume hallucination on error

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "hallucination_rate": float(avg_score)
    }
