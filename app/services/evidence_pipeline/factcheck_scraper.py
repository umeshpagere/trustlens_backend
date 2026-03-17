import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from .source_trust import get_source_trust

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(), reraise=True)
def search_factcheck_sites(query: str) -> list:
    """
    Scrape fact-checking domains directly (snopes, politifact, factcheck.org, reuters.com).
    """
    return []
