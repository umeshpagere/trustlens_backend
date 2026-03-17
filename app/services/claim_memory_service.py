"""
TrustLens Claim Memory & Similarity Engine — Part-13

Stores verified claim results in MongoDB and performs cosine-based
vector search to detect semantically similar previously-verified claims.

Re-verification is skipped when a stored claim is found with:
  cosine_similarity >= 0.88
  confidence        >= 0.70
  aligned_sentences >= 5
  coverage_score    >= 0.50

Memory entries expire automatically after 180 days via a TTL index.

MongoDB collection: claim_memory (in trustlensDB)
Vector search index name: claim_memory_index  (create in Atlas UI — see Step 9)
"""

import logging
from datetime import datetime, timezone, timedelta

try:
    from pymongo import MongoClient, ASCENDING
    import certifi
except Exception:
    MongoClient = None
    certifi = None

from app.config.settings import Config
from app.utils.text_utils import normalize_claim_text
from app.models.model_loader import get_embedding_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD   = 0.88   # cosine similarity required for memory hit
MIN_CONFIDENCE         = 0.70   # minimum confidence to store/reuse
MIN_ALIGNED_SENTENCES  = 5      # minimum evidence sentences to store/reuse
MIN_COVERAGE_SCORE     = 0.50   # minimum retrieval coverage to store/reuse
MEMORY_TTL_DAYS        = 180    # entries older than this are expired by MongoDB

# ---------------------------------------------------------------------------
# MongoDB lazy singleton
# ---------------------------------------------------------------------------
_mongo_client = None
_collection   = None


def _get_collection():
    """
    Lazily initialise and return the claim_memory MongoDB collection.
    Mirrors the pattern used in analysis_storage_service.py.
    On first call also ensures the TTL and vector indexes exist.
    """
    global _mongo_client, _collection

    if _collection is not None:
        return _collection

    if MongoClient is None:
        logger.warning("[ClaimMemory] pymongo not installed — memory disabled")
        return None

    if not Config.MONGODB_URI:
        logger.warning("[ClaimMemory] MONGODB_URI not set — memory disabled")
        return None

    try:
        allow_invalid = getattr(Config, "MONGODB_TLS_ALLOW_INVALID_CERTIFICATES", False)
        connect_args = {
            "connect": True,
            "serverSelectionTimeoutMS": 10000,
            "socketTimeoutMS": 10000,
            "tls": True,
            "tlsAllowInvalidCertificates": allow_invalid,
        }
        if certifi and not allow_invalid:
            connect_args["tlsCAFile"] = certifi.where()

        _mongo_client = MongoClient(Config.MONGODB_URI, **connect_args)
        db = _mongo_client[Config.MONGODB_DATABASE]
        col = db["claim_memory"]

        # TTL index — Step 10: auto-expire after 180 days
        col.create_index(
            [("created_at", ASCENDING)],
            expireAfterSeconds=MEMORY_TTL_DAYS * 86400,
            name="claim_memory_ttl",
            background=True,
        )

        # Dedup index on normalized claim text (exact match fast-path)
        col.create_index(
            [("normalized_claim", ASCENDING)],
            name="claim_memory_normalized",
            background=True,
        )

        _collection = col
        logger.info("[ClaimMemory] Connected to MongoDB claim_memory collection")
        return _collection

    except Exception as exc:
        logger.error(f"[ClaimMemory] MongoDB connection failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 4 — Embedding generation
# ---------------------------------------------------------------------------
def generate_embedding(text: str) -> list:
    """
    Generate a 384-dim float32 embedding using the preloaded
    all-MiniLM-L6-v2 model.  Returns a plain Python list for MongoDB storage.
    Returns [] on failure.
    """
    try:
        model = get_embedding_model()
        if model is None:
            return []
        vec = model.encode([text])[0]
        return vec.tolist()
    except Exception as exc:
        logger.warning(f"[ClaimMemory] Embedding generation failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Step 5 — Memory lookup via Atlas Vector Search
# ---------------------------------------------------------------------------
def find_similar_claim(claim_text: str, similarity_threshold: float = SIMILARITY_THRESHOLD) -> dict | None:
    """
    Look up a semantically similar claim in the memory collection.

    1. Normalise the input claim
    2. Fast exact-match check (avoids LLM embedding cost)
    3. Generate embedding
    4. Run $vectorSearch aggregate against claim_memory_index
    5. Filter by similarity_threshold and quality thresholds

    Returns the stored memory document if a sufficient match is found,
    otherwise returns None.
    """
    col = _get_collection()
    if col is None:
        return None

    norm = normalize_claim_text(claim_text)
    if not norm:
        return None

    # Fast path: exact normalized text match
    try:
        exact = col.find_one({"normalized_claim": norm})
        if exact:
            stored = exact
            if _is_reusable(stored):
                logger.info(f"[ClaimMemory] Exact hit for: '{norm[:60]}'")
                return _format_result(stored)
    except Exception as exc:
        logger.warning(f"[ClaimMemory] Exact lookup failed: {exc}")

    # Vector search path
    query_vector = generate_embedding(norm)
    if not query_vector:
        return None

    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index":        "claim_memory_index",
                    "path":         "embedding",
                    "queryVector":  query_vector,
                    "numCandidates": 50,
                    "limit":        1,
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
        ]
        results = list(col.aggregate(pipeline))

        if not results:
            logger.info(f"[ClaimMemory] Miss (no candidates): '{norm[:60]}'")
            return None

        top = results[0]
        score = top.get("score", 0.0)

        if score < similarity_threshold:
            logger.info(f"[ClaimMemory] Miss (score={score:.3f} < {similarity_threshold}): '{norm[:60]}'")
            return None

        if not _is_reusable(top):
            logger.info(f"[ClaimMemory] Hit but not reusable (quality gates failed): '{norm[:60]}'")
            return None

        logger.info(f"[ClaimMemory] Vector hit (score={score:.3f}): '{norm[:60]}'")
        return _format_result(top)

    except Exception as exc:
        # Atlas vector search index may not exist yet — degrade gracefully
        logger.warning(f"[ClaimMemory] Vector search failed (index may not be created yet): {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 6 — Store a verified claim
# ---------------------------------------------------------------------------
def store_claim_result(claim_text: str, verification: dict, stats: dict) -> bool:
    """
    Store a verified claim in the memory collection if quality gates pass.

    Safety rules:
      - confidence        >= 0.70
      - aligned_sentences >= 5
      - coverage_score    >= 0.50
      - verdict must not be "ERROR"

    Lightweight fields only — agent_outputs and reasoning_steps are excluded.
    Returns True if stored, False otherwise.
    """
    col = _get_collection()
    if col is None:
        return False

    confidence        = float(verification.get("confidence", 0.0))
    aligned_sentences = int(stats.get("aligned_sentences", 0))
    coverage_score    = float(stats.get("coverage_score", 0.0))
    verdict           = str(verification.get("verdict", "UNVERIFIED"))

    if confidence < MIN_CONFIDENCE:
        logger.debug(f"[ClaimMemory] Skip store — confidence too low ({confidence:.2f})")
        return False
    if aligned_sentences < MIN_ALIGNED_SENTENCES:
        logger.debug(f"[ClaimMemory] Skip store — insufficient aligned sentences ({aligned_sentences})")
        return False
    if coverage_score < MIN_COVERAGE_SCORE:
        logger.debug(f"[ClaimMemory] Skip store — coverage too low ({coverage_score:.2f})")
        return False
    if verdict == "ERROR":
        logger.debug("[ClaimMemory] Skip store — verdict is ERROR")
        return False

    norm = normalize_claim_text(claim_text)
    if not norm:
        return False

    embedding = generate_embedding(norm)
    if not embedding:
        logger.warning("[ClaimMemory] Skip store — embedding failed")
        return False

    doc = {
        "claim_text":        claim_text,
        "normalized_claim":  norm,
        "embedding":         embedding,
        "verdict":           verdict,
        "credibility_score": int(verification.get("credibility_score", 50)),
        "confidence":        confidence,
        "trusted_sources":   verification.get("trusted_sources", []),
        "source_agreement":  float(verification.get("source_agreement", 0.0)),
        "coverage_score":    coverage_score,
        "aligned_sentences": aligned_sentences,
        "created_at":        datetime.now(timezone.utc),
    }

    try:
        col.replace_one({"normalized_claim": norm}, doc, upsert=True)
        logger.info(
            f"[ClaimMemory] Stored: '{norm[:60]}' "
            f"verdict={verdict} confidence={confidence:.2f}"
        )
        return True
    except Exception as exc:
        logger.warning(f"[ClaimMemory] Store failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_reusable(doc: dict) -> bool:
    """Check that a memory document meets reuse quality gates."""
    return (
        float(doc.get("confidence", 0))        >= MIN_CONFIDENCE         and
        int(doc.get("aligned_sentences", 0))   >= MIN_ALIGNED_SENTENCES  and
        float(doc.get("coverage_score", 0))    >= MIN_COVERAGE_SCORE
    )


def _format_result(doc: dict) -> dict:
    """
    Produce a pipeline-compatible result dict from a memory document.
    Shape matches what _process_single_claim() returns so the rest of
    analyze.py can merge it transparently.
    """
    return {
        "claim":  doc.get("claim_text", ""),
        "source": "memory",
        "verification": {
            "verdict":           doc.get("verdict", "UNVERIFIED"),
            "credibility_score": doc.get("credibility_score", 50),
            "confidence":        doc.get("confidence", 0.5),
            "explanation":       "Result reused from claim memory (high-confidence prior verification).",
            "trusted_sources":   doc.get("trusted_sources", []),
            "source_agreement":  doc.get("source_agreement", 0.0),
            "reasoning_steps":   [],
        },
        "evidence":       [],
        "retrieval_meta": {},
        "stats": {
            "retrieved_docs":    0,
            "aligned_sentences": doc.get("aligned_sentences", 0),
            "coverage_score":    doc.get("coverage_score", 0.0),
            "retrieval_loops":   0,
        },
        "_memory_reuse": True,
    }
