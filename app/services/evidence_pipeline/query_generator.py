

def generate_queries(claim: str, entities: list = None) -> list:
    """
    Generate a diverse set of search queries by combining:
    - the raw claim
    - entity-specific news queries
    - entity pair queries
    - a fact-check oriented query

    Falls back gracefully when no entities are provided.
    """
    if entities is None:
        entities = []

    queries = []

    # 1. Base claim as-is
    queries.append(claim)

    # 2. Entity-specific news queries
    for entity in entities:
        queries.append(f"{entity} news")

    # 3. Top-2 entity combination query
    if len(entities) >= 2:
        combo = " ".join(entities[:2])
        queries.append(combo)

    # 4. Fact-check oriented query
    queries.append(f"{claim} fact check")

    # Deduplicate while preserving insertion order
    seen = set()
    unique_queries = []
    for q in queries:
        normalized = q.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_queries.append(normalized)

    return unique_queries[:5]
