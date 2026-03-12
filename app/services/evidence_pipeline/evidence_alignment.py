import logging

logger = logging.getLogger(__name__)

import nltk
from nltk.tokenize import sent_tokenize
from .semantic_ranker import rank_evidence

# Ensure punkt is downloaded lazily
_punkt_downloaded = False


def get_sentences(text: str) -> list:
    global _punkt_downloaded
    if not _punkt_downloaded:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            _punkt_downloaded = True
        except Exception:
            pass
    try:
        return sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return []


def align_evidence(claim: str, evidence_items: list, use_nli: bool = True) -> list:
    """
    Takes a list of evidence objects and returns the top-5 aligned sentences.
    Each sentence inherits the trust score of its parent evidence item.
    """
    all_sentence_items = []
    for item in evidence_items:
        content = item.get("text", "")
        trust_score = item.get("trust_score", 0.5)
        source = item.get("source", "Unknown")
        
        sentences = get_sentences(content)
        for s in sentences:
            all_sentence_items.append({
                "text": s,
                "trust_score": trust_score,
                "source": source
            })

    if not all_sentence_items:
        return []

    # rank_evidence must be updated to handle list of dicts
    ranked_sentences = rank_evidence(claim, all_sentence_items, use_nli=use_nli)

    # Deduplicate while preserving order
    seen = set()
    unique_ranked = []
    for rs in ranked_sentences:
        if rs["text"] not in seen:
            seen.add(rs["text"])
            unique_ranked.append(rs)
            if len(unique_ranked) == 5:
                break
    return unique_ranked
