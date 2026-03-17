
from typing import List


def _compress_claim(claim: str) -> str:
    """
    Lightweight keyword compression: drop very common stopwords and
    keep signal-bearing tokens in order.
    """
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "by",
        "at",
        "with",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "been",
        "will",
        "would",
        "could",
        "should",
        "today",
        "yesterday",
        "tomorrow",
    }
    tokens = claim.split()
    filtered = [t for t in tokens if t.lower() not in stopwords]
    return " ".join(filtered)


def generate_queries(claim: str, entities: List[str] | None = None) -> list:
    """
    Generate structured, high-relevance search queries for a factual claim.

    Required query set per claim:
      1. Exact claim
      2. Entity + event/action
      3. Entity + object
      4. Event-focused description
      5. Fact-check framed query
      6. Compressed keyword query

    This utility is used by the evaluation harness and legacy pipeline
    paths; the adaptive retrieval planner has its own expansion logic.
    """

    if not claim:
        return []

    entities = entities or []
    primary_entity = entities[0] if entities else ""

    queries: list[str] = []

    # 1. Exact claim
    queries.append(claim.strip())

    # Simple heuristic split: first verb-ish token as action pivot
    tokens = claim.split()
    action_idx = None
    for i, tok in enumerate(tokens):
        lower = tok.lower().strip(",.")
        if lower in {
            "bought",
            "buys",
            "buying",
            "purchased",
            "purchase",
            "declared",
            "announced",
            "said",
            "won",
            "lost",
            "kills",
            "killed",
            "died",
            "arrested",
            "appointed",
            "elected",
        }:
            action_idx = i
            break

    if action_idx is not None:
        action_phrase = " ".join(tokens[action_idx : action_idx + 3])
        object_phrase = " ".join(tokens[action_idx + 1 : action_idx + 8])
    else:
        action_phrase = ""
        object_phrase = " ".join(tokens[1:8])

    # 2. Entity + event/action
    if primary_entity and action_phrase:
        queries.append(f"{primary_entity} {action_phrase}")

    # 3. Entity + object
    if primary_entity and object_phrase:
        queries.append(f"{primary_entity} {object_phrase}")

    # 4. Event-focused description (claim without leading entity, if present)
    if primary_entity and claim.startswith(primary_entity):
        event_desc = claim[len(primary_entity) :].strip(" ,.-")
        if event_desc:
            queries.append(f"{primary_entity} {event_desc}")

    # 5. Fact-check framed query
    queries.append(f"{claim} fact check")
    if primary_entity:
        queries.append(f"{primary_entity} {action_phrase or 'claim'} fact check".strip())

    # 6. Compressed keyword query
    compressed = _compress_claim(claim)
    if compressed:
        queries.append(compressed)

    # Fallback safety: ensure we always have at least 3 queries
    if len(queries) < 3:
        queries.append(f"{claim} evidence")
        queries.append(_compress_claim(claim + " news"))

    # Deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        q_clean = q.strip()
        if q_clean and q_clean not in seen:
            seen.add(q_clean)
            unique_queries.append(q_clean)

    # Cap at a reasonable number for legacy paths
    return unique_queries[:8]
