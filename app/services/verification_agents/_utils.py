"""
Shared utilities for the verification_agents module.

Provides:
  - GLOBAL_SAFETY_RULES: injected into every agent system prompt
  - extract_json(): robust JSON extraction from LLM responses
"""

import json
import re
import logging
import asyncio
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple Async Memory Cache (In-Memory)
# ---------------------------------------------------------------------------
class AsyncCache:
    """
    Thread-safe in-memory async cache with TTL expiry and oldest-first eviction.

    Parameters
    ----------
    max_size : int
        Maximum number of entries before the oldest is evicted.
    ttl_seconds : float
        How long (in seconds) an entry is considered fresh. A ``get()`` call
        for an expired entry will delete it and return ``None``.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 300):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None
            if time.monotonic() - self._timestamps[key] > self._ttl:
                # Entry has expired — delete and treat as a miss
                del self._cache[key]
                del self._timestamps[key]
                return None
            return self._cache[key]

    async def set(self, key: str, value: Any):
        async with self._lock:
            if len(self._cache) >= self._max_size:
                # Evict the entry with the oldest insertion timestamp
                oldest_key = min(self._timestamps, key=lambda k: self._timestamps[k])
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            self._cache[key] = value
            self._timestamps[key] = time.monotonic()


# Global cache instance for claim analysis (5-minute TTL)
CLAIM_ANALYSIS_CACHE = AsyncCache(max_size=200, ttl_seconds=300)

# ---------------------------------------------------------------------------
# Global safety block injected at the top of every agent system prompt
# ---------------------------------------------------------------------------
GLOBAL_SAFETY_RULES = """\
You are part of a fact-checking system.

Your job is NOT to guess missing information.
You must ONLY reason using the information provided in the input.

STRICT RULES:
1. Do not invent facts, sources, dates, or events.
2. Do not use outside knowledge beyond what is provided.
3. Only reason using the provided claim and evidence.
4. If the information is insufficient, return "INSUFFICIENT_EVIDENCE" for that field.
5. If the information is ambiguous, return "UNCERTAIN" for that field.
6. Never fabricate entities, locations, or events.
7. Your response must be valid JSON with no markdown fences.

Return JSON only.
"""


# ---------------------------------------------------------------------------
# Robust JSON extractor
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    """
    Extract a JSON object from an LLM response, tolerating:
      - ```json ... ``` fences
      - Leading/trailing prose
      - Minor whitespace issues
    """
    text = text.strip()
    # Strip common markdown fences
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*",     "", text)
    text = re.sub(r"\s*```$",     "", text)
    text = text.strip()

    # Fast path: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Regex fallback: grab first complete {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"No valid JSON found in LLM response. "
        f"Preview: {text[:200]}"
    )
