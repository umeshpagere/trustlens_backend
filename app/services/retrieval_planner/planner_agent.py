"""
Retrieval Planner Agent — Part-12

LLM agent that decides:
  • which sources to query
  • how many queries to generate
  • retrieval depth (light / medium / deep)

It does NOT verify the claim and does NOT invent evidence.
"""

import logging
from app.config.settings import Config
from app.config.azure import get_async_azure_client
from app.services.verification_agents._utils import extract_json, GLOBAL_SAFETY_RULES

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------
MAX_QUERIES_CAP = 8           # hard cap regardless of planner output
ALL_SOURCES = [
    "news_api",
    "factcheck_api",
    "wikipedia",
    "web_search",
    "rss_feeds",
    "factcheck_scraper",
]

SYSTEM_PROMPT = f"""\
{GLOBAL_SAFETY_RULES}

You are a retrieval planning agent inside a fact-checking system.

Your ONLY job is to decide HOW to retrieve evidence for a claim.

AVAILABLE SOURCES:
  news_api          — recent news articles (EventRegistry)
  factcheck_api     — Google FactCheck database
  wikipedia         — encyclopaedic / historical facts
  web_search        — general web results for niche topics
  rss_feeds         — syndicated news feeds
  factcheck_scraper — scrapes known fact-check websites

ROUTING RULES:
1. Choose the MINIMUM sources necessary to verify the claim.
2. If the claim resembles known misinformation → include factcheck_api + factcheck_scraper.
3. If the claim is about a recent or breaking event → prioritise news_api + rss_feeds + web_search.
4. If the claim is historical or encyclopaedic → prioritise wikipedia.
5. If the claim is niche or obscure → include web_search.
6. Always include at least 2 sources.

OUTPUT RULES:
• max_queries must be between 3 and 8.
• retrieval_depth must be: "light" | "medium" | "deep"
  - light  → max_queries ≤ 4, target_documents ≤ 20
  - medium → max_queries ≤ 6, target_documents ≤ 40
  - deep   → max_queries ≤ 8, target_documents ≤ 70
• priority_entities: entities ranked by how relevant they are for search.
• Do NOT invent sources not listed above.
• Do NOT verify the claim.
• Return JSON only.
"""

USER_TEMPLATE = """\
Claim:
{claim}

Claim Analysis:
  entities: {entities}
  event_type: {event_type}
  temporal_signal: {temporal_signal}
  expected_evidence_types: {expected_evidence_types}

Return ONLY valid JSON:
{{
  "sources": ["<subset of: news_api, factcheck_api, wikipedia, web_search, rss_feeds, factcheck_scraper>"],
  "max_queries": <int 3-8>,
  "retrieval_depth": "light | medium | deep",
  "priority_entities": ["<most search-relevant entities>"],
  "target_documents": <int 20-70>
}}
"""

# --- Fallback plan (used when LLM fails) ------------------------------------
def _default_plan(event_type: str = "UNKNOWN") -> dict:
    """Deterministic fallback plan covering all sources at medium depth."""
    if event_type in ("political", "military", "health", "crisis"):
        sources = ["news_api", "factcheck_api", "rss_feeds", "factcheck_scraper"]
    elif event_type in ("historical", "scientific", "geographic"):
        sources = ["wikipedia", "factcheck_api", "web_search"]
    else:
        sources = ALL_SOURCES
    return {
        "sources":          sources,
        "max_queries":      5,
        "retrieval_depth":  "medium",
        "priority_entities": [],
        "target_documents": 40,
    }


async def plan_retrieval(
    claim: str,
    claim_meta: dict,
    entities: list = None,
) -> dict:
    """
    Agent 0 (planner): produce a retrieval strategy for the claim.

    Args:
        claim:      raw claim text
        claim_meta: output from analyze_claim() (entities, event_type, temporal_signal, ...)
        entities:   optional entity list override (from extract_entities)

    Returns:
        dict with keys: sources, max_queries, retrieval_depth, priority_entities, target_documents
    """
    _entities    = entities or claim_meta.get("entities", [])
    event_type   = claim_meta.get("event_type", "UNKNOWN")
    temp_signal  = claim_meta.get("temporal_signal", "UNKNOWN")
    ev_types     = claim_meta.get("expected_evidence_types", [])
    fallback     = _default_plan(event_type)

    try:
        client = get_async_azure_client()
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    claim=claim,
                    entities=_entities,
                    event_type=event_type,
                    temporal_signal=temp_signal,
                    expected_evidence_types=ev_types,
                )},
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        plan = extract_json(raw)

        # Validate / sanitise LLM output
        plan["sources"]      = [s for s in plan.get("sources", []) if s in ALL_SOURCES] or fallback["sources"]
        plan["max_queries"]  = max(3, min(MAX_QUERIES_CAP, int(plan.get("max_queries", 5))))
        plan["retrieval_depth"] = plan.get("retrieval_depth", "medium") if plan.get("retrieval_depth") in ("light", "medium", "deep") else "medium"
        plan["target_documents"] = max(10, min(70, int(plan.get("target_documents", 40))))

        logger.info(
            f"[PlannerAgent] sources={plan['sources']} "
            f"max_queries={plan['max_queries']} depth={plan['retrieval_depth']}"
        )
        return plan

    except Exception as exc:
        logger.warning(f"[PlannerAgent] Failed ({exc}), using default plan")
        return fallback
