import logging
from app.config.settings import Config
from app.utils.domain_utils import extract_and_normalize_domain
from .source_trust import get_source_trust

logger = logging.getLogger(__name__)


def search_web(query: str) -> list:
    """
    Performs a web search via Tavily Search API.
    Returns standard evidence structure.
    Requires TAVILY_API_KEY in .env
    
    Tavily is an AI-optimized search engine designed for LLM/RAG applications,
    providing clean structured snippets ideal for fact-checking.
    """
    if not Config.TAVILY_API_KEY:
        logger.warning("[web_search] TAVILY_API_KEY not set — skipping web search")
        return []

    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=Config.TAVILY_API_KEY)
        
        # Tavily search with optimized parameters for fact-checking
        response = client.search(
            query=query,
            max_results=10,
            search_depth="basic",  # "basic" is faster, "advanced" for deeper search
            include_answer=False,   # We don't need AI summary, just sources
            include_raw_content=False,  # Don't need full page content
        )
        
        logger.info(f"[web_search] Tavily search completed for query='{query[:80]}'")
        
        results = response.get("results", [])
        items = []
        
        for r in results:
            url_str = r.get("url", "")
            content = r.get("content", "")
            
            if not content:
                continue
                
            items.append({
                "source": r.get("title", "Web"),
                "domain": extract_and_normalize_domain(url_str),
                "type": "web",
                "text": content,
                "title": r.get("title", ""),
                "url": url_str,
                "published_at": r.get("published_date"),
                "trust_score": get_source_trust(url_str, "web"),
                "relevance_score": r.get("score", 0.0),  # Tavily's relevance score
            })
        
        logger.info(f"[web_search] Retrieved {len(items)} results for '{query[:80]}'")
        return items
    
    except ImportError:
        logger.error("[web_search] tavily-python not installed. Run: pip install tavily-python")
        return []
    except Exception as exc:
        # Tavily SDK raises generic exceptions for auth errors, quota exhaustion, etc.
        error_msg = str(exc).lower()
        
        if "api key" in error_msg or "unauthorized" in error_msg:
            logger.error("[web_search] Invalid TAVILY_API_KEY — verify key in .env")
        elif "quota" in error_msg or "limit" in error_msg:
            logger.warning("[web_search] Tavily API quota exhausted (1000/month free tier)")
        else:
            logger.error(f"[web_search] Tavily search failed for query='{query[:80]}': {type(exc).__name__}: {exc}")
        
        return []
