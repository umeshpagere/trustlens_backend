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

# ---------------------------------------------------------------------------
# NLI Model Preloading
# ---------------------------------------------------------------------------
print("⚙️ Loading NLI verification model (cross-encoder/nli-roberta-base)...")
try:
    _nli_pipeline = pipeline(
        "text-classification",
        model="cross-encoder/nli-roberta-base",   # lighter alternative to roberta-large-mnli
        device=-1  # CPU; switch to 0 for GPU
    )
    print("✅ NLI verification model loaded")
except Exception as e:
    print(f"❌ Failed to load NLI model: {e}")
    logger.error(f"Failed to load NLI model: {e}")
    _nli_pipeline = None

def get_nli_pipeline():
    """Access the preloaded NLI singleton."""
    try:
        if _nli_pipeline is None:
            logger.warning("NLI pipeline requested but it is None")
        return _nli_pipeline
    except Exception:
        logger.exception("NLI pipeline not available")
        raise


def check_contradiction(claim: str, evidence: str) -> dict:
    """
    Run Natural Language Inference between a claim and an evidence snippet.

    Returns a dict with:
        label  : "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL"
        score  : float (confidence of that label)
    """
    nli = get_nli_pipeline()
    if not nli:
        return {"label": "NEUTRAL", "score": 0.0}

    try:
        # Standard NLI format: premise </s></s> hypothesis
        sequence = f"{evidence} </s></s> {claim}"
        result = nli(sequence, truncation=True, max_length=512)[0]

        # Normalise label to upper-case
        label = result["label"].upper()

        # Some models return LABEL_0/LABEL_1/LABEL_2 — map them
        label_map = {
            "LABEL_0": "CONTRADICTION",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "ENTAILMENT",
        }
        label = label_map.get(label, label)

        return {
            "label": label,
            "score": float(result["score"])
        }
    except Exception:
        logger.exception("Error in NLI check")
        return {"label": "NEUTRAL", "score": 0.0}


def compute_nli_score(nli_result: dict) -> float:
    """
    Convert an NLI result into a single float score suitable for ranking:
       ENTAILMENT   → +score
       CONTRADICTION → -score
       NEUTRAL      →  0.0
    """
    label = nli_result.get("label", "NEUTRAL")
    score = nli_result.get("score", 0.0)

    if label == "ENTAILMENT":
        return score
    elif label == "CONTRADICTION":
        return -score
    return 0.0
