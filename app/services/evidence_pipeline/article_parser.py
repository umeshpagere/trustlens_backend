import logging
from newspaper import Article

logger = logging.getLogger(__name__)

def parse_article_text(url: str) -> str:
    """
    Downloads and extracts the main text body from an article URL using newspaper3k.
    Returns empty string on failure.
    """
    if not url:
        return ""
        
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.warning(f"Failed to parse article at {url}: {e}")
        return ""
