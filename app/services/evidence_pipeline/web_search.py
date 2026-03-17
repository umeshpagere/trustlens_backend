import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from .source_trust import get_source_trust

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(), reraise=True)
def search_web(query: str) -> list:
    """
    Performs a web search for the query (e.g. via SerpAPI or Bing).
    Returns standard evidence structure.
    """
    # Assuming the API returns a list of results with title, link, snippet
    return []
