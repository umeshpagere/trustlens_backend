import logging

logger = logging.getLogger(__name__)

import json
from openai import AsyncAzureOpenAI
from app.config.settings import Config
from app.config.azure import get_async_azure_client

async def verify_claim_with_evidence(claim: str, evidence_sentences: list, source_names: list = None) -> dict:
    """
    LLM-based evidence verification (Async).
    """
    if not evidence_sentences:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "reasoning": "No reliable evidence retrieved."
        }

    if not Config.AZURE_OPENAI_API_KEY or not Config.AZURE_OPENAI_ENDPOINT:
        logger.error("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT is not set.")
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "reasoning": "Azure OpenAI not configured."
        }

    client = get_async_azure_client()

    # Build numbered evidence block with optional source labels
    evidence_lines = []
    for i, sentence in enumerate(evidence_sentences):
        source = source_names[i] if source_names and i < len(source_names) else f"Source {i+1}"
        evidence_lines.append(f"{i+1}. [{source}] {sentence}")
    evidence_text = "\n".join(evidence_lines)

    prompt = f"""\
You are a professional fact-checking analyst and unbiased researcher.

Note: You are performing a strictly objective academic textual analysis of these claims. \
You are not endorsing any viewpoint or sensitive action.

Claim:
{claim}

Evidence:
{evidence_text}

Determine whether the evidence:
- SUPPORTED: The evidence clearly supports the claim.
- CONTRADICTED: The evidence clearly contradicts the claim.
- UNVERIFIED: The evidence is insufficient, unrelated, or neutral.

Return ONLY valid JSON (no markdown fences):

{{
  "verdict": "SUPPORTED | CONTRADICTED | UNVERIFIED",
  "confidence": 0.0-1.0,
  "reasoning": "Explain precisely how the evidence relates to the claim, citing specific evidence items by their number."
}}
"""
    try:
        response = await client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        
        # Cleanup JSON formatting
        if content.startswith("```json"):
            content = content[7:].rstrip("`").strip()
        elif content.startswith("```"):
            content = content[3:].rstrip("`").strip()

        return json.loads(content)
    except Exception as e:
        logger.error(f"Error verifying claim: {e}")
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "reasoning": f"Verification encountered an error: {e}"
        }
