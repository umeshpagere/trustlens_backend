import os
import pandas as pd
import logging
from ragas import evaluate
from ragas.metrics import faithfulness
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RagasTest")

def test_ragas():
    load_dotenv()
    
    # Debug env vars
    logger.info(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    logger.info(f"AZURE_OPENAI_API_KEY present: {bool(os.getenv('AZURE_OPENAI_API_KEY'))}")
    
    # Workaround for Ragas/LangChain internal requirements
    if os.getenv('AZURE_OPENAI_API_KEY'):
        os.environ["OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT')
        logger.info("Internal environment variables set for Ragas.")
    
    from datasets import Dataset
    data = {
        "question": ["What is TrustLens?"],
        "answer": ["TrustLens is a misinformation detection platform."],
        "contexts": [["TrustLens is a project designed to analyze social media content for credibility and factual accuracy using AI and evidence retrieval."]],
        "ground_truth": ["TrustLens is a misinformation detection and credibility analysis tool."]
    }
    dataset = Dataset.from_dict(data)
    
    logger.info("Dataset (datasets.Dataset) prepared for Ragas.")

    # 2. Configure Azure LLM
    # Use environment variables directly to ensure no config-layer issues
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    logger.info("Azure OpenAI client for Ragas initialized.")

    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    
    try:
        # 3. Run evaluation with all metrics
        logger.info("Starting Ragas evaluation (all 4 metrics)...")
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm
        )
        
        print("\n=== Ragas Test Result ===")
        print(result)
        print("========================\n")
        
    except Exception as e:
        logger.error(f"Ragas standalone test failed: {e}")
        # Print more details if it's a connection error
        if "api_key" in str(e).lower():
            logger.error("Check your AZURE_OPENAI_API_KEY in .env")

if __name__ == "__main__":
    test_ragas()
