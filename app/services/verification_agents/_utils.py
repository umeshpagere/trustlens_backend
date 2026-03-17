"""
Shared utilities for the verification_agents module.

Provides:
  - GLOBAL_SAFETY_RULES: injected into every agent system prompt
  - extract_json(): robust JSON extraction from LLM responses
"""

import json
import re
import logging

logger = logging.getLogger(__name__)

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
