import json
import os
import logging

logger = logging.getLogger(__name__)

def load_dataset(path):
    """
    Loads the evaluation dataset from a JSON file.
    Validates that the file exists and contains a list of samples.
    """
    if not os.path.exists(path):
        logger.error(f"Dataset file not found at: {path}")
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error("Dataset format error: Expected a list of samples.")
            raise ValueError("Dataset must be a JSON list.")
            
        logger.info(f"Successfully loaded {len(data)} samples from {path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse dataset JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}")
        raise
