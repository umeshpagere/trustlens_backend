import os
import logging
from deepeval.models import AzureOpenAIModel
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepEvalTest")

def test_deepeval():
    load_dotenv()
    
    logger.info(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    logger.info(f"AZURE_OPENAI_API_KEY present: {bool(os.getenv('AZURE_OPENAI_API_KEY'))}")
    
    try:
        model = AzureOpenAIModel(
            model="gpt-4o-mini",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        logger.info("DeepEval Azure model initialized.")

        metric = HallucinationMetric(threshold=0.5, model=model)
        test_case = LLMTestCase(
            input="What is TrustLens?",
            actual_output="TrustLens is a misinformation detection platform.",
            retrieval_context=["TrustLens is a project designed to analyze social media content for credibility."],
            context=["TrustLens is a project designed to analyze social media content for credibility."]
        )
        
        logger.info("Starting DeepEval measurement...")
        metric.measure(test_case)
        print(f"\nHallucination Score: {metric.score}")
        print(f"Reason: {metric.reason}\n")
        
    except Exception as e:
        logger.error(f"DeepEval standalone test failed: {e}")

if __name__ == "__main__":
    test_deepeval()
