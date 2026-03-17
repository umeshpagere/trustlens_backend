import logging
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import AzureChatOpenAI
from app.config.settings import Config

logger = logging.getLogger(__name__)

def evaluate_with_ragas(eval_data):
    """
    Evaluates a set of samples using ragas metrics.
    
    Args:
        eval_data (list[dict]): List of samples, each with:
            - question: str (the claim)
            - answer: str (the agent's explanation)
            - contexts: list[str] (retrieved evidence)
            - ground_truth: str (the ground truth verdict)
            
    Returns:
        dict: Average scores for faithfulness, answer_relevancy, context_precision, context_recall.
    """
    if not eval_data:
        return {}

    logger.info(f"Running Ragas evaluation on {len(eval_data)} samples")

    try:
        from datasets import Dataset
        import os
        # Ensure environment variables are set for Ragas/LangChain internal usage
        os.environ["AZURE_OPENAI_API_KEY"] = Config.AZURE_OPENAI_API_KEY
        os.environ["AZURE_OPENAI_ENDPOINT"] = Config.AZURE_OPENAI_ENDPOINT
        # Some versions of LangChain/OpenAI implicitly require this even for Azure
        os.environ["OPENAI_API_KEY"] = Config.AZURE_OPENAI_API_KEY
        
        # Prepare the dataset for Ragas
        dataset = Dataset.from_list(eval_data)
        
        from langchain_openai import AzureChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Configure the Azure OpenAI model for Ragas
        llm = AzureChatOpenAI(
            azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
        )
        
        # Use the same embedding model as the production pipeline for consistency
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Run evaluation
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=llm,
            embeddings=embeddings
        )
        
        return result.to_pandas().mean(numeric_only=True).to_dict()

    except Exception as e:
        logger.error(f"Ragas evaluation failed: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "error": str(e)
        }
