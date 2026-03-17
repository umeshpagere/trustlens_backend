"""
Query Expander — Part-12

Replaces the static 5-template query_generator.py with a smarter
expansion strategy:

  1. Base claim (always included)
  2. Entity-focused queries (per priority entity from planner)
  3. Fact-check framed queries
  4. Temporal queries (if claim has temporal signal)
  5. LLM-generated semantic variants (async, optional)

Total query count is bounded by planner_output["max_queries"] (≤ 8).
"""

import logging
import re
from app.config.settings import Config
from app.config.azure import get_async_azure_client
from app.services.verification_agents._utils import extract_json

logger = logging.getLogger(__name__)


def _compress_query(text: str) -> str:
    stop = {
        "the", "a", "an", "of", "in", "on", "for", "to", "by", "at", "with", "that", "this",
        "these", "those", "is", "are", "was", "were", "has", "have", "had", "be", "been",
        "being", "will", "would", "could", "should",
    }
    tokens = re.findall(r"[A-Za-z0-9₹$€£]+", text or "")
    tokens = [t for t in tokens if t.lower() not in stop]
    return " ".join(tokens)


def _extract_numbers_and_dates(text: str) -> list[str]:
    patterns = [
        r"\b(19\d{2}|20\d{2})\b",
        r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\b",
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\s+\d{1,2}\b",
        r"\b₹\s?\d+(?:\.\d+)?\s?(?:crore|cr|lakh|million|billion)?\b",
        r"\b\d+(?:\.\d+)?\s?(?:crore|cr|lakh|million|billion|%)\b",
    ]
    found: list[str] = []
    for p in patterns:
        for m in re.findall(p, text or "", flags=re.IGNORECASE):
            val = m if isinstance(m, str) else " ".join(m)
            if val and val not in found:
                found.append(val)
    return found[:4]


def _guess_action_phrase(claim: str) -> str:
    toks = (claim or "").split()
    if not toks:
        return ""
    verbs = {
        "bought", "purchased", "acquired", "sold", "announced", "declared", "said", "claimed",
        "reported", "won", "lost", "killed", "arrested", "appointed", "elected", "resigned",
        "approved", "banned",
    }
    for i, t in enumerate(toks):
        if t.lower().strip(",.") in verbs:
            return " ".join(toks[i:i + 3]).strip()
    return ""


def _build_template_queries(
    claim: str,
    entities: list,
    priority_entities: list,
    temporal_signal: str,
    max_queries: int,
) -> list:
    """
    Fast, deterministic template-based expansion.
    Always runs even if LLM expansion fails.
    """
    queries = []

    relevant_entities = (priority_entities or entities)[:3]
    primary_entity = relevant_entities[0] if relevant_entities else ""
    action = _guess_action_phrase(claim)
    numbers = _extract_numbers_and_dates(claim)
    compressed = _compress_query(claim)

    # Structured query set (aim >= 6), avoiding generic “entity news” placeholders.
    # 1) Exact claim
    queries.append(claim)

    # 2) Entity + event/action (+ numbers if present)
    if primary_entity and action:
        q = f"{primary_entity} {action}"
        if numbers:
            q = f"{q} {' '.join(numbers)}"
        queries.append(q)

    # 3) Entity + object (compressed tokens)
    if primary_entity and compressed:
        queries.append(f"{primary_entity} {compressed}")

    # 4) Event description
    if action and compressed:
        queries.append(f"{action} {compressed}".strip())

    # 5) Fact-check framed query
    if compressed:
        queries.append(f"{compressed} fact check".strip())
    else:
        queries.append(f"{claim} fact check")
    if primary_entity:
        queries.append(f"{primary_entity} {(action or compressed or 'claim')} fact check".strip())

    # 6) Keyword compressed query
    if compressed and compressed != claim:
        queries.append(compressed)

    # Temporal marker inclusion (extra)
    if temporal_signal and temporal_signal not in ("UNKNOWN", "none"):
        queries.append(f"{compressed or claim} {temporal_signal}".strip())

    # Deduplicate, preserving order
    seen = set()
    unique = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            unique.append(q)
        if len(unique) >= max_queries:
            break

    return unique


async def _llm_query_expansion(
    claim: str,
    entities: list,
    max_extras: int,
) -> list:
    """
    Optional LLM step: generate max_extras semantic query variants.
    Returns [] on any failure — never blocks the controller.
    """
    if max_extras <= 0:
        return []

    prompt = f"""\
Generate {max_extras} distinct search queries to retrieve evidence about this claim.
Each query should be short (≤ 10 words), specific, and searchable.
Do not rephrase the claim — diversify the angle.

Claim: {claim}
Key entities: {entities}

Return ONLY valid JSON:
{{
  "queries": ["<query 1>", "<query 2>", ...]
}}
"""
    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        raw  = response.choices[0].message.content.strip()
        data = extract_json(raw)
        expanded = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        logger.info(f"[QueryExpander] LLM generated {len(expanded)} extra queries")
        return expanded[:max_extras]
    except Exception as exc:
        logger.warning(f"[QueryExpander] LLM expansion failed: {exc}")
        return []


async def expand_queries(
    claim: str,
    entities: list,
    planner_output: dict,
    temporal_signal: str = "UNKNOWN",
) -> list:
    """
    Produce an expanded, deduplicated list of search queries.

    Args:
        claim:          raw claim text
        entities:       entity list from extract_entities()
        planner_output: from plan_retrieval()
        temporal_signal: from claim_meta

    Returns:
        list of query strings, len ≤ planner_output["max_queries"]
    """
    max_queries       = planner_output.get("max_queries", 5)
    priority_entities = planner_output.get("priority_entities", [])

    # Phase 1: deterministic templates (fast, always available)
    template_queries = _build_template_queries(
        claim, entities, priority_entities, temporal_signal, max_queries
    )

    # Phase 2: LLM semantic expansion (fills remaining slots)
    remaining = max_queries - len(template_queries)
    if remaining > 0 and Config.AZURE_OPENAI_API_KEY:
        llm_extras = await _llm_query_expansion(claim, entities, remaining)
    else:
        llm_extras = []

    # Merge and deduplicate
    seen = set(template_queries)
    final = list(template_queries)
    for q in llm_extras:
        if q not in seen and len(final) < max_queries:
            seen.add(q)
            final.append(q)

    logger.info(
        f"[QueryExpander] {len(final)} queries "
        f"(templates={len(template_queries)} llm={len(llm_extras)}) "
        f"cap={max_queries}"
    )
    return final
