import logging

logger = logging.getLogger(__name__)

import nltk
from nltk.tokenize import sent_tokenize
from .semantic_ranker import rank_evidence
from .evidence_filter import classify_evidence

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
        published_at = item.get("published_at")
        
        sentences = get_sentences(content)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean.split()) < 5 or len(s_clean) < 20:
                continue
            if not any(c.isalpha() for c in s_clean):
                continue
                
            all_sentence_items.append({
                "text": s_clean,
                "trust_score": trust_score,
                "source": source,
                "domain": item.get("domain", "unknown"),
                "timestamp": published_at,
                "url": item.get("url", ""),
                "title": item.get("title", "")
            })

    if not all_sentence_items:
        return []

    logger.info(f"Sentences extracted: {len(all_sentence_items)}")
    # rank_evidence must be updated to handle list of dicts
    ranked_sentences = rank_evidence(claim, all_sentence_items, use_nli=use_nli)
    logger.info(f"Sentences ranked: {len(ranked_sentences)}")

    # AI Filtering, Diversity Control, and Balanced Selection
    seen = set()
    source_counts = {}    # max 2 sentences per source display name
    domain_counts = {}    # max 3 sentences per domain (Part-9)
    
    supporting = []
    contradicting = []
    
    for rs in ranked_sentences:
        if rs["text"] in seen:
            continue
            
        # Diversity control: max 2 per source display name
        source_name = rs.get("source", "Unknown").lower()
        if source_counts.get(source_name, 0) >= 2:
            continue

        # Domain diversity: max 3 sentences per domain (Part-9)
        domain_name = rs.get("domain", "unknown")
        if domain_counts.get(domain_name, 0) >= 3:
            continue
            
        if use_nli:
            nli_result = classify_evidence(claim, rs["text"])
            rs["nli_label"] = nli_result["label"]
            rs["nli_score"] = nli_result["confidence"]
        else:
            rs["nli_label"] = "IRRELEVANT"
            rs["nli_score"] = 0.0

        label = rs.get("nli_label", "IRRELEVANT")

        # STRICT FILTERING:
        #   - Drop IRRELEVANT evidence entirely from downstream verification.
        #   - If the NLI model is unavailable, classify_evidence returns UNKNOWN.
        #     We keep UNKNOWN evidence sparingly to avoid total recall collapse,
        #     but it should not dominate the evidence set.
        if label == "IRRELEVANT":
            continue

        if label == "SUPPORTED":
            if len(supporting) < 4:
                supporting.append(rs)
                seen.add(rs["text"])
                source_counts[source_name] = source_counts.get(source_name, 0) + 1
                domain_counts[domain_name] = domain_counts.get(domain_name, 0) + 1
        elif label == "CONTRADICTED":
            if len(contradicting) < 4:
                contradicting.append(rs)
                seen.add(rs["text"])
                source_counts[source_name] = source_counts.get(source_name, 0) + 1
                domain_counts[domain_name] = domain_counts.get(domain_name, 0) + 1
        elif label == "UNKNOWN":
            # Conservative fallback: keep at most 2 UNKNOWN items total.
            # This preserves some evidence for verification when NLI is offline,
            # while still prioritizing properly labeled items.
            unknown_count = sum(1 for _x in (supporting + contradicting) if _x.get("nli_label") == "UNKNOWN")
            if unknown_count < 2:
                supporting.append(rs)  # treat as weak-support bucket for ordering only
                seen.add(rs["text"])
                source_counts[source_name] = source_counts.get(source_name, 0) + 1
                domain_counts[domain_name] = domain_counts.get(domain_name, 0) + 1

        if len(supporting) == 4 and len(contradicting) == 4:
            break

    # Combine in SCORE ORDER (not grouped by label) to avoid positional bias.
    # Re-sort all accepted sentences by (semantic) score descending.
    unique_ranked = sorted(
        supporting + contradicting,
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    logger.info(
        f"Filtered evidence sentences: {len(unique_ranked)} "
        f"(Supported: {len(supporting)}, Contradicted: {len(contradicting)})"
    )
    return unique_ranked
