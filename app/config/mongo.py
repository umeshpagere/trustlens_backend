import os
import certifi
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)

_mongo_client = None

def get_mongo_client() -> MongoClient:
    """
    Returns a shared, lazily-initialized MongoClient instance.
    This prevents creating a new connection pool for every service
    that needs to talk to MongoDB.
    """
    global _mongo_client
    if _mongo_client is None:
        from app.config.settings import Config
        mongo_uri = Config.MONGODB_URI
        if not mongo_uri:
            raise ValueError("MONGODB_URI environment variable not set.")
            
        try:
            allow_invalid = getattr(Config, "MONGODB_TLS_ALLOW_INVALID_CERTIFICATES", False)
            tls_args = {
                "tls": True,
                "tlsAllowInvalidCertificates": allow_invalid
            }
            if certifi and not allow_invalid:
                tls_args["tlsCAFile"] = certifi.where()
                
            _mongo_client = MongoClient(mongo_uri, **tls_args)
            logger.info("✅ Initialized shared MongoDB client (tlsAllowInvalid: %s)", allow_invalid)
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
            
    return _mongo_client
