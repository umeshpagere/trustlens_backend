"""
TrustLens Narrative Intelligence Engine — Part-13

Tracks misinformation narratives and claim clusters across the system.

Responsibilities:
  1. Cluster semantically similar claims (cosine ≥ 0.85 on centroids)
  2. Compute misinformation ratio per cluster
  3. Detect coordinated misinformation campaigns
  4. Build a claim relationship graph
  5. Return narrative metadata for the API response

MongoDB collections:
  - narrative_clusters  (cluster centroids + stats)
  - claim_graph         (pairwise claim similarity edges)

Atlas Vector Search index (create manually in Atlas UI):
  Collection: narrative_clusters
  Index name: narrative_cluster_index
  Field: centroid_embedding  |  numDimensions: 384  |  similarity: cosine

All Mongo ops are synchronous (pymongo) and must be wrapped in
asyncio.to_thread() when called from async context.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config.settings import Config
from app.config.mongo import get_mongo_client
from app.utils.text_utils import normalize_claim_text
from app.models.model_loader import get_embedding_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds & constants
# ---------------------------------------------------------------------------
CLUSTER_SIMILARITY_THRESHOLD = 0.85   # centroid cosine similarity for cluster match
GRAPH_EDGE_THRESHOLD         = 0.80   # pairwise similarity for graph edges
MIN_CONFIDENCE               = 0.70   # skip claims below this confidence
MIN_COVERAGE_SCORE           = 0.50
MIN_ALIGNED_SENTENCES        = 5

CAMPAIGN_MIN_SIZE            = 15     # cluster must have ≥ this many claims
CAMPAIGN_MISINFO_RATIO       = 0.70   # and ≥ this misinformation ratio

MISINFO_HIGH_THRESHOLD       = 0.70
MISINFO_MEDIUM_THRESHOLD     = 0.40

# ---------------------------------------------------------------------------
# MongoDB lazy singletons — one for each collection
# ---------------------------------------------------------------------------
_mongo_client     = None
_clusters_col     = None
_graph_col        = None


def _get_clusters_col():
    global _mongo_client, _clusters_col
    if _clusters_col is not None:
        return _clusters_col
    col = _init_collection("narrative_clusters")
    if col is not None:
        # TTL index: keep clusters for 6 months
        try:
            col.create_index([("last_updated", ASCENDING)],
                             expireAfterSeconds=180 * 86400,
                             name="narrative_ttl", background=True)
        except Exception:
            pass
    _clusters_col = col
    return _clusters_col


def _get_graph_col():
    global _graph_col
    if _graph_col is not None:
        return _graph_col
    col = _init_collection("claim_graph")
    if col is not None:
        try:
            col.create_index([("timestamp", ASCENDING)],
                             expireAfterSeconds=180 * 86400,
                             name="graph_ttl", background=True)
            col.create_index([("claim_a", ASCENDING), ("claim_b", ASCENDING)],
                             name="graph_pair", background=True)
        except Exception:
            pass
    _graph_col = col
    return _graph_col


def _init_collection(name: str):
    if not Config.MONGODB_URI:
        logger.warning("[NarrativeEngine] MONGODB_URI not set — narrative tracking disabled")
        return None
    try:
        client = get_mongo_client()
        db = client[Config.MONGODB_DATABASE]
        logger.info(f"[NarrativeEngine] Connected to MongoDB collection: {name}")
        return db[name]
    except Exception as exc:
        logger.error(f"[NarrativeEngine] MongoDB connection failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------
def embed_claim(text: str) -> list:
    """
    Generate a 384-dim float32 embedding.
    Returns plain Python list for MongoDB storage.
    Uses the same preloaded model as claim_memory_service.
    """
    try:
        model = get_embedding_model()
        if model is None:
            return []
        vec = model.encode([text])[0]
        return vec.tolist()
    except Exception as exc:
        logger.warning(f"[NarrativeEngine] Embedding failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Misinformation ratio helpers
# ---------------------------------------------------------------------------
def _compute_misinfo_ratio(supported: int, contradicted: int, unverified: int) -> float:
    total = supported + contradicted + unverified
    if total == 0:
        return 0.0
    return round(contradicted / total, 4)


def _misinfo_risk_level(ratio: float) -> str:
    if ratio >= MISINFO_HIGH_THRESHOLD:
        return "HIGH"
    if ratio >= MISINFO_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Cluster operations (sync — call via asyncio.to_thread from async context)
# ---------------------------------------------------------------------------
def find_similar_cluster(embedding: list) -> Optional[dict]:
    """
    Run $vectorSearch on narrative_cluster_index.
    Returns the matching cluster document or None.
    """
    col = _get_clusters_col()
    if col is None or not embedding:
        return None
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index":         "narrative_cluster_index",
                    "path":          "centroid_embedding",
                    "queryVector":   embedding,
                    "numCandidates": 50,
                    "limit":         1,
                }
            },
            {"$addFields": {"_vscore": {"$meta": "vectorSearchScore"}}},
        ]
        results = list(col.aggregate(pipeline))
        if results and results[0].get("_vscore", 0) >= CLUSTER_SIMILARITY_THRESHOLD:
            return results[0]
        return None
    except Exception as exc:
        # Index may not exist yet — degrade gracefully
        logger.warning(f"[NarrativeEngine] Vector search failed (index may be missing): {exc}")
        return None


def create_cluster(claim_text: str, embedding: list, verdict: str) -> str:
    """
    Create a new narrative cluster seeded by this claim.
    Returns the new cluster_id.
    """
    col = _get_clusters_col()
    if col is None:
        return ""

    cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)

    supported    = 1 if verdict == "SUPPORTED"    else 0
    contradicted = 1 if verdict == "CONTRADICTED" else 0
    unverified   = 1 if verdict == "UNVERIFIED"   else 0
    ratio        = _compute_misinfo_ratio(supported, contradicted, unverified)

    doc = {
        "cluster_id":            cluster_id,
        "representative_claim":  claim_text,
        "centroid_embedding":    embedding,
        "claim_count":           1,
        "supported_claims":      supported,
        "contradicted_claims":   contradicted,
        "unverified_claims":     unverified,
        "misinformation_ratio":  ratio,
        "campaign_detected":     False,
        "risk_level":            _misinfo_risk_level(ratio),
        "first_seen":            now,
        "last_updated":          now,
    }
    try:
        col.insert_one(doc)
        logger.info("[NarrativeEngine] New cluster created: %s", cluster_id)
    except Exception as exc:
        logger.warning(f"[NarrativeEngine] Cluster insert failed: {exc}")

    return cluster_id


def update_cluster(cluster: dict, embedding: list, verdict: str) -> dict:
    """
    Add a claim to an existing cluster:
      - incrementally update verdict counters
      - recalculate centroid as online mean
      - recompute misinformation ratio
      - flag campaign if thresholds met
    Returns updated cluster metadata dict.
    """
    col = _get_clusters_col()
    if col is None:
        return cluster

    cluster_id   = cluster["cluster_id"]
    old_count    = int(cluster.get("claim_count", 1))
    new_count    = old_count + 1

    # Incremental centroid update: c_new = (c_old * n + v) / (n + 1)
    old_centroid = np.array(cluster.get("centroid_embedding", []), dtype=np.float32)
    new_vec      = np.array(embedding, dtype=np.float32)
    if old_centroid.shape == new_vec.shape:
        new_centroid = ((old_centroid * old_count) + new_vec) / new_count
    else:
        new_centroid = new_vec
    new_centroid_list = new_centroid.tolist()

    supported    = cluster.get("supported_claims",    0) + (1 if verdict == "SUPPORTED"    else 0)
    contradicted = cluster.get("contradicted_claims", 0) + (1 if verdict == "CONTRADICTED" else 0)
    unverified   = cluster.get("unverified_claims",   0) + (1 if verdict == "UNVERIFIED"   else 0)
    ratio        = _compute_misinfo_ratio(supported, contradicted, unverified)

    campaign = (new_count >= CAMPAIGN_MIN_SIZE and ratio >= CAMPAIGN_MISINFO_RATIO)
    if campaign and not cluster.get("campaign_detected"):
        logger.warning(
            "[NarrativeEngine] Campaign detected: cluster=%s ratio=%.2f size=%d",
            cluster_id, ratio, new_count
        )

    now = datetime.now(timezone.utc)
    update = {
        "$set": {
            "centroid_embedding":  new_centroid_list,
            "claim_count":         new_count,
            "supported_claims":    supported,
            "contradicted_claims": contradicted,
            "unverified_claims":   unverified,
            "misinformation_ratio": ratio,
            "risk_level":          _misinfo_risk_level(ratio),
            "campaign_detected":   campaign,
            "last_updated":        now,
        }
    }
    try:
        col.update_one({"cluster_id": cluster_id}, update)
        logger.info(
            "[NarrativeEngine] Cluster updated: cluster_id=%s size=%d ratio=%.2f",
            cluster_id, new_count, ratio
        )
    except Exception as exc:
        logger.warning(f"[NarrativeEngine] Cluster update failed: {exc}")

    # Return updated metadata (without re-reading from DB for speed)
    return {
        **cluster,
        "claim_count":          new_count,
        "supported_claims":     supported,
        "contradicted_claims":  contradicted,
        "unverified_claims":    unverified,
        "misinformation_ratio": ratio,
        "campaign_detected":    campaign,
        "risk_level":           _misinfo_risk_level(ratio),
    }


# ---------------------------------------------------------------------------
# Claim graph edge creation (sync)
# ---------------------------------------------------------------------------
def store_graph_edge(claim_a: str, claim_b: str, similarity: float,
                     verdict_a: str, verdict_b: str) -> None:
    """
    Insert a pairwise similarity edge into claim_graph.
    Only called when similarity >= GRAPH_EDGE_THRESHOLD.
    """
    col = _get_graph_col()
    if col is None:
        return
    doc = {
        "claim_a":     claim_a,
        "claim_b":     claim_b,
        "similarity":  round(float(similarity), 4),
        "verdict_pair": [verdict_a, verdict_b],
        "timestamp":   datetime.now(timezone.utc),
    }
    try:
        col.insert_one(doc)
    except Exception as exc:
        logger.debug(f"[NarrativeEngine] Graph edge insert failed: {exc}")


# ---------------------------------------------------------------------------
# Main entry point — called from analyze.py (async)
# ---------------------------------------------------------------------------
async def process_narratives(evidence_results: list) -> list:
    """
    Process all verified claims through the narrative engine.

    Args:
        evidence_results: list of _process_single_claim() result dicts

    Returns:
        list of per-claim narrative metadata dicts (one per input, or {} if skipped)
    """
    import asyncio

    narrative_summaries = []

    # Filter and extract valid claims
    valid_claims = []
    for idx, result in enumerate(evidence_results):
        verification = result.get("verification", {})
        stats        = result.get("stats", {})
        confidence   = float(verification.get("confidence", 0.0))
        cov_score    = float(stats.get("coverage_score", 0.0))
        aligned      = int(stats.get("aligned_sentences", 0))
        verdict      = str(verification.get("verdict", "UNVERIFIED"))
        raw_text     = result.get("claim", "")

        if (confidence < MIN_CONFIDENCE or
                cov_score < MIN_COVERAGE_SCORE or
                aligned < MIN_ALIGNED_SENTENCES):
            continue

        normalized = normalize_claim_text(raw_text)
        if normalized:
            valid_claims.append((idx, normalized, verdict))

    if not valid_claims:
        # Pre-fill empty narratives for all
        return [{}] * len(evidence_results)

    # Parallelize embeddings
    embed_coros = [asyncio.to_thread(embed_claim, c[1]) for c in valid_claims]
    embeddings = await asyncio.gather(*embed_coros)

    # Prepare batch loop state
    processed_this_batch: list[tuple[str, list, str]] = []
    narrative_summaries = [{}] * len(evidence_results)

    # We still do cluster updates sequentially to avoid write races on the same centroid
    for i, (orig_idx, normalized, verdict) in enumerate(valid_claims):
        embedding = embeddings[i]
        if not embedding:
            continue

        cluster = await asyncio.to_thread(find_similar_cluster, embedding)

        if cluster:
            cluster = await asyncio.to_thread(update_cluster, cluster, embedding, verdict)
            cluster_id = cluster["cluster_id"]
        else:
            cluster_id = await asyncio.to_thread(create_cluster, normalized, embedding, verdict)
            ratio = 1.0 if verdict == "CONTRADICTED" else 0.0
            cluster = {
                "cluster_id":           cluster_id,
                "claim_count":          1,
                "misinformation_ratio": ratio,
                "campaign_detected":    False,
                "risk_level":           _misinfo_risk_level(ratio),
            }

        # Graph edges — compare against claims already processed in this batch
        for prev_text, prev_embedding, prev_verdict in processed_this_batch:
            try:
                sim = cosine_similarity([embedding], [prev_embedding])[0][0]
                if sim >= GRAPH_EDGE_THRESHOLD:
                    await asyncio.to_thread(
                        store_graph_edge,
                        normalized, prev_text, sim, verdict, prev_verdict
                    )
            except Exception as exc:
                logger.debug(f"[NarrativeEngine] Graph edge computation error: {exc}")

        processed_this_batch.append((normalized, embedding, verdict))

        narrative_summaries[orig_idx] = {
            "cluster_id":           cluster.get("cluster_id", cluster_id),
            "cluster_size":         cluster.get("claim_count", 1),
            "misinformation_ratio": cluster.get("misinformation_ratio", 0.0),
            "risk_level":           cluster.get("risk_level", "LOW"),
            "campaign_detected":    cluster.get("campaign_detected", False),
        }

    return narrative_summaries
