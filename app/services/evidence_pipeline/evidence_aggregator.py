import logging
import requests
import re
import asyncio
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config.settings import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from .source_trust import get_source_trust
from app.utils.domain_utils import extract_and_normalize_domain

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(), reraise=True)
def search_wikipedia(query: str) -> list:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "prop": "revisions|info",
        "rvprop": "timestamp",
        "inprop": "url",
        "utf8": "1",
        "formatversion": "2"
    }
    headers = {
        "User-Agent": "EvidencePipeline/1.0"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        search_pages = data.get("query", {}).get("pages", [])
        
        results = []
        for page in search_pages:
            snippet = page.get("extract", "") or page.get("title", "")
            snippet = re.sub(r'<[^>]+>', '', snippet)
            revisions = page.get("revisions", [])
            published_at = revisions[0]["timestamp"] if revisions else None
            page_url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{page.get('title', '').replace(' ', '_')}")
            results.append({
                "source": "Wikipedia",
                "domain": extract_and_normalize_domain(page_url),
                "type": "knowledge",
                "text": snippet,
                "title": page.get("title", ""),
                "url": page_url,
                "published_at": published_at,
                "trust_score": get_source_trust(page_url, "knowledge")
            })
        return results
    except Exception:
        logger.exception("Error fetching from Wikipedia API")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(), reraise=True)
def search_factcheck(query: str) -> list:
    if not Config.GOOGLE_FACTCHECK_API_KEY:
        logger.error("GOOGLE_FACTCHECK_API_KEY is not set.")
        return []
    
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query,
        "key": Config.GOOGLE_FACTCHECK_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        claims = data.get("claims", [])
        results = []
        for claim in claims:
            reviewer = claim.get("claimReview", [{}])[0]
            publisher = reviewer.get("publisher", {}).get("name", "Unknown Publisher")
            verdict = reviewer.get("textualRating", "No Rating")
            review_url = reviewer.get("url", "")
            published_at = claim.get("claimDate") or reviewer.get("reviewDate")
            results.append({
                "source": publisher,
                "domain": extract_and_normalize_domain(review_url),
                "type": "fact_check",
                "claim_review": verdict,
                "text": f"{publisher} rated the claim: {verdict}. Context: {claim.get('text', '')}",
                "url": review_url,
                "published_at": published_at,
                "trust_score": get_source_trust(review_url, "fact_check")
            })
        return results
    except Exception:
        logger.exception("Error fetching from Fact Check API")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(), reraise=True)
def search_news(query: str) -> list:
    if not Config.NEWS_API_KEY:
        logger.error("NEWS_API_KEY is not set.")
        return []
        
    url = "http://eventregistry.org/api/v1/article/getArticles"
    payload = {
        "action": "getArticles",
        "keyword": query,
        "articlesPage": 1,
        "articlesCount": 20,
        "articlesSortBy": "rel",
        "articlesSortByAsc": False,
        "resultType": "articles",
        "apiKey": Config.NEWS_API_KEY
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        articles = data.get("articles", {}).get("results", [])
        
        results = []
        for article in articles:
            source_name = article.get("source", {}).get("title", "Unknown News")
            article_url = article.get("url", "")
            published_at = article.get("dateTimePub") or article.get("date")
            results.append({
                "source": source_name,
                "domain": extract_and_normalize_domain(article_url),
                "type": "news",
                "text": article.get("body", "") or article.get("title", ""),
                "title": article.get("title", ""),
                "url": article_url,
                "published_at": published_at,
                "trust_score": get_source_trust(article_url, "news")
            })
        return results
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        if status in [429, 503]:
            logger.warning(f"Event Registry rate limit/concurrency exceeded ({status}). Skipping news retrieval.")
        elif status == 401:
            logger.warning("Event Registry unauthorized (401) - Check API Key. Skipping news retrieval.")
        else:
            logger.exception(f"HTTP Error {status} fetching from Event Registry")
        return []
    except Exception:
        logger.exception("Error fetching from Event Registry")
        return []

async def aggregate_evidence(queries: list, sources: list = None) -> list:
    """
    Retrieves and aggregates evidence from multiple sources in parallel.
    Uses up to 8 queries (adaptive, set by planner) and deduplicates results.

    Args:
        queries:  list of search query strings
        sources:  optional allowlist of source names from:
                  ["news_api", "factcheck_api", "wikipedia",
                   "web_search", "rss_feeds", "factcheck_scraper"]
                  If None, all sources are queried (backward-compatible).
    """
    from .web_search import search_web
    from .rss_retriever import search_rss
    from .factcheck_scraper import search_factcheck_sites

    # Source name → callable mapping (Part-12 source routing)
    SOURCE_MAP = {
        "news_api":          search_news,
        "factcheck_api":     search_factcheck,
        "wikipedia":         search_wikipedia,
        "web_search":        search_web,
        "rss_feeds":         search_rss,
        "factcheck_scraper": search_factcheck_sites,
    }

    if not queries:
        return []

    # Resolve which sources to call
    if sources:
        active_sources = {k: v for k, v in SOURCE_MAP.items() if k in sources}
    else:
        active_sources = SOURCE_MAP   # backward-compatible: all sources

    # Limit to maximum 8 queries per claim (planner cap)
    used_queries = queries[:8]
    logger.info(f"Queries executed: {len(used_queries)} | Sources: {list(active_sources.keys())}")

    all_evidence = []
    seen_urls = set()
    counts = {"fact_check": 0, "news": 0, "knowledge": 0, "web": 0, "rss": 0, "scrape": 0}

    tasks = []

    # 1. Parallelize Retrieval across all queries and active sources
    for query in used_queries:
        for fn in active_sources.values():
            tasks.append(asyncio.to_thread(fn, query))
        
    # Execute batch
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 2. Extract, Enforce Limits, and Deduplicate
    all_results = []
    for res in results:
        if isinstance(res, Exception):
            logger.error(f"Retrieval task failed: {res}")
            continue
        if res:
            all_results.extend(res)
            
    # Group by type to enforce limits per type per total retrieval
    factcheck_items = [r for r in all_results if r["type"] == "fact_check"]
    wikipedia_items = [r for r in all_results if r["type"] == "knowledge"]
    news_items = [r for r in all_results if r["type"] == "news"]
    web_items = [r for r in all_results if r.get("source") == "web" or r.get("type") == "web"]
    rss_items = [r for r in all_results if r.get("type") == "rss"]
    scrape_items = [r for r in all_results if r.get("type") == "factcheck_scrape"]
    
    # Retrieval depth targets:
    #   - FactCheck API: up to 10
    #   - Wikipedia API: up to 10
    #   - News API:      up to 20
    #   - Other sources: kept but secondary
    final_items = (
        factcheck_items[:10] +
        wikipedia_items[:10] +
        news_items[:20] +
        web_items[:20] +
        rss_items[:10] +
        scrape_items[:10]
    )

    # Document Deduplication
    for item in final_items:
        url = item.get("url", "")
        text = item.get("text", "")
        
        # Fallback if URL is missing: use deterministic MD5 (hash() is PYTHONHASHSEED-randomized)
        identifier = url if url else hashlib.md5(text.encode("utf-8")).hexdigest()
        
        if identifier not in seen_urls:
            seen_urls.add(identifier)
            all_evidence.append(item)
            type_key = item.get("type", "unknown")
            if type_key in counts:
                counts[type_key] += 1

    logger.info(f"Evidence retrieved: {len(all_evidence)} | Fact-check: {counts['fact_check']} | News: {counts['news']} | Knowledge: {counts['knowledge']}")
    logger.info(f"Documents deduplicated: {len(final_items) - len(all_evidence)}")
    
    return all_evidence
