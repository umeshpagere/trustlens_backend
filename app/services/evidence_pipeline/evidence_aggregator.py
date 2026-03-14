import logging
import requests
import re
from app.config.settings import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from .source_trust import get_source_trust

logger = logging.getLogger(__name__)

def search_wikipedia(query: str) -> list:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
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
        search_results = data.get("query", {}).get("search", [])
        
        results = []
        for item in search_results:
            snippet = item.get("snippet", "")
            snippet = re.sub(r'<[^>]+>', '', snippet)
            results.append({
                "source": "Wikipedia",
                "type": "knowledge",
                "text": snippet,
                "title": item.get("title", ""),
                "url": f"https://en.wikipedia.org/wiki/{item.get('title', '').replace(' ', '_')}",
                "trust_score": get_source_trust("Wikipedia", "knowledge")
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching from Wikipedia API: {e}")
        return []

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
            results.append({
                "source": publisher,
                "type": "fact_check",
                "claim_review": verdict,
                "text": f"{publisher} rated the claim: {verdict}. Context: {claim.get('text', '')}",
                "url": reviewer.get("url", ""),
                "trust_score": get_source_trust(publisher, "fact_check")
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching from Fact Check API: {e}")
        return []

def search_news(query: str) -> list:
    if not Config.NEWS_API_KEY:
        logger.error("NEWS_API_KEY is not set.")
        return []
        
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": Config.NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        
        results = []
        for article in articles:
            source_name = article.get("source", {}).get("name", "Unknown News")
            results.append({
                "source": source_name,
                "type": "news",
                "text": article.get("content", "") or article.get("description", ""),
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "trust_score": get_source_trust(source_name, "news")
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching from News API: {e}")
        return []

def aggregate_evidence(queries: list) -> list:
    """
    Retrieves and aggregates evidence from multiple sources.
    Returns a unified list of evidence objects.
    """
    if not queries:
        return []

    # Use the first query as the primary search term
    query = queries[0]
    all_evidence = []

    # Parallelize API calls
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_search = {
            executor.submit(search_factcheck, query): "factcheck",
            executor.submit(search_wikipedia, query): "wikipedia",
            executor.submit(search_news, query): "news"
        }
        
        counts = {"fact_check": 0, "news": 0, "knowledge": 0}
        
        for future in as_completed(future_to_search):
            try:
                data = future.result()
                name = future_to_search[future]
                
                if name == "factcheck":
                    for item in data[:3]:
                        all_evidence.append(item)
                        counts["fact_check"] += 1
                elif name == "wikipedia":
                    for item in data[:3]:
                        all_evidence.append(item)
                        counts["knowledge"] += 1
                elif name == "news":
                    for item in data[:5]:
                        all_evidence.append(item)
                        counts["news"] += 1
            except Exception as e:
                logger.error(f"Parallel search error: {e}")

    logger.info(f"Evidence retrieved: {len(all_evidence)} | Fact-check sources: {counts['fact_check']} | News sources: {counts['news']} | Knowledge sources: {counts['knowledge']}")
    return all_evidence
