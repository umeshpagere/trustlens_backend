import os
import logging
import warnings
from transformers import pipeline

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

print("⚙️ Loading NLI evidence filter model (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)...")
try:
    _evidence_filter = pipeline(
        "text-classification",
        model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device=-1
    )
    print("✅ Evidence filter model loaded")
except Exception as e:
    print(f"❌ Failed to load evidence filter model: {e}")
    logger.critical(
        "NLI evidence filter model failed to load. All evidence will be classified "
        "as IRRELEVANT — verification quality will be severely degraded. "
        "Run pre_download_models.py during Docker build to fix this. Error: %s", e
    )
    _evidence_filter = None

def get_evidence_filter():
    return _evidence_filter

def classify_evidence(claim: str, sentence: str) -> dict:
    """
    Classify whether the evidence sentence supports, contradicts, or is irrelevant to the claim.
    Returns:
        {
            "label": "SUPPORTED" | "CONTRADICTED" | "IRRELEVANT",
            "confidence": 0.0-1.0
        }
    """
    filter_pipe = get_evidence_filter()
    if not filter_pipe:
        # Fallback: do not hard-discard everything when the NLI model is unavailable.
        # Downstream alignment will treat UNKNOWN conservatively.
        return {"label": "UNKNOWN", "confidence": 0.0}

    try:
        # Many NLI models expect premise then hypothesis. Let's provide claim and sentence.
        # NLI format: sentence (premise) -> claim (hypothesis)
        sequence = f"{sentence} </s></s> {claim}"
        result = filter_pipe(sequence, truncation=True, max_length=512)[0]
        
        label = result["label"].upper()
        
        # Mapping standard MNLI labels
        label_map = {
            "ENTAILMENT": "SUPPORTED",
            "CONTRADICTION": "CONTRADICTED",
            "NEUTRAL": "IRRELEVANT"
        }
        mapped_label = label_map.get(label, "UNKNOWN")
        
        return {
            "label": mapped_label,
            "confidence": float(result["score"])
        }
    except Exception:
        logger.exception("Error in AI evidence filtering")
        return {"label": "UNKNOWN", "confidence": 0.0}


def classify_evidence_batch(claim: str, sentences: list[str], batch_size: int = 8) -> list[dict]:
    """
    Batch classify multiple sentences against a claim for improved performance.
    
    This reduces NLI processing time from ~90s (50 sentences × 1.8s each) to ~15-20s
    by processing sentences in batches instead of one-by-one.
    
    Args:
        claim: The claim to verify
        sentences: List of evidence sentences to classify
        batch_size: Number of sentences to process per batch (default 8)
    
    Returns:
        List of dicts with "label" and "confidence" for each sentence
    """
    filter_pipe = get_evidence_filter()
    if not filter_pipe:
        # Fallback: return UNKNOWN for all sentences
        return [{"label": "UNKNOWN", "confidence": 0.0}] * len(sentences)
    
    if not sentences:
        return []
    
    try:
        # Create all sequences at once
        sequences = [f"{sent} </s></s> {claim}" for sent in sentences]
        
        # Batch inference - HuggingFace pipeline supports batch processing
        # This is much faster than calling the pipeline 50 times individually
        results = filter_pipe(sequences, truncation=True, max_length=512, batch_size=batch_size)
        
        # Map results to our label format
        label_map = {
            "ENTAILMENT": "SUPPORTED",
            "CONTRADICTION": "CONTRADICTED",
            "NEUTRAL": "IRRELEVANT"
        }
        
        mapped_results = []
        for result in results:
            label = result["label"].upper()
            mapped_results.append({
                "label": label_map.get(label, "UNKNOWN"),
                "confidence": float(result["score"])
            })
        
        logger.info(f"Batch NLI: classified {len(sentences)} sentences in batches of {batch_size}")
        return mapped_results
        
    except Exception as e:
        logger.exception(f"Error in batch NLI filtering, falling back to single-item processing: {e}")
        # Fallback: process one-by-one if batch fails
        return [classify_evidence(claim, sent) for sent in sentences]
