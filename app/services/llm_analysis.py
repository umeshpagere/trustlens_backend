import json
import re
import base64
import asyncio
import logging
from datetime import datetime, timezone
from app.config.azure import get_azure_client
from app.config.settings import Config
from app.prompts.claim_extraction_prompt import CLAIM_EXTRACTION_SYSTEM_PROMPT
from app.prompts.image_claim_extraction_prompt import IMAGE_CLAIM_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 8000


def safe_parse_json(raw: str) -> dict:
    """Parse JSON, recovering gracefully from truncated LLM responses."""
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Recovery: walk back from the end to find a parseable prefix
    for end in range(len(raw), 0, -1):
        try:
            return json.loads(raw[:end] + "}")
        except json.JSONDecodeError:
            continue
    # Last resort: extract first {...} block
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}

def _sanitize_user_input(text: str) -> str:
    """Sanitize user input to prevent prompt injection and preserve JSON safety."""
    if not text or not isinstance(text, str):
        return ""
    sanitized = text.strip()
    if len(sanitized) > MAX_INPUT_LENGTH:
        sanitized = sanitized[:MAX_INPUT_LENGTH] + "... [truncated]"
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", sanitized)
    return sanitized

def detect_image_mime_type(image_bytes: bytes) -> str:
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    elif image_bytes[:2] == b'\xff\xd8':
        return "image/jpeg"
    elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    else:
        return "image/jpeg"

async def extract_claims_from_text(text: str) -> list:
    """Extract verifiable factual claims from text."""
    sanitized_text = _sanitize_user_input(text)
    
    messages = [
        {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract verifiable claims from this content:\n\n{sanitized_text}"},
    ]

    client = get_azure_client()

    def _sdk_call() -> str:
        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    try:
        content = await asyncio.to_thread(_sdk_call)
        result = json.loads(content.strip())
        raw_claims = result.get("claims", [])
        
        structured_claims = []
        for c in raw_claims:
            if isinstance(c, str):
                structured_claims.append({
                    "text": c,
                    "temporal_signal": None,
                    "explicit_date": None
                })
            elif isinstance(c, dict):
                structured_claims.append({
                    "text": c.get("text", str(c)),
                    "temporal_signal": c.get("temporal_signal"),
                    "explicit_date": c.get("explicit_date")
                })
        return structured_claims
    except Exception as e:
        logger.error(f"Text claim extraction failed: {e}")
        return []

async def analyze_text_with_llm(text: str) -> dict:
    """Performs full semantic analysis and claim extraction from text."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        raise ValueError("Text input is required")
    
    sanitized_text = _sanitize_user_input(text)
    
    # Combined prompt: claim extraction + semantic analysis in one call.
    # NOTE: We do NOT reuse CLAIM_EXTRACTION_SYSTEM_PROMPT here because it
    # contains "Do NOT generate explanations" which conflicts with our need
    # for reasoningSummary, primaryClaim, etc.
    combined_prompt = (
        "You are a factual claim extraction and semantic analysis engine.\n\n"
        "TASK 1 — CLAIM EXTRACTION:\n"
        "Extract up to 5 verifiable factual claims from the content.\n"
        "A valid claim MUST have a clear subject + specific action/event.\n"
        "REFORMULATE any 'the paper/article says...' into direct factual statements.\n"
        "Skip opinions, emotional language, vague descriptions, and speculation.\n\n"
        "TASK 2 — SEMANTIC ANALYSIS:\n"
        "Analyze the text for credibility signals:\n"
        "- Identify the single most important factual claim (primaryClaim).\n"
        "- Score overall semantic credibility from 0-100 (semanticScore). "
        "High score = credible, low = suspicious.\n"
        "- Rate your confidence from 0.0-1.0 (confidenceScore).\n"
        "- List any manipulation indicators (emotional appeals, misleading framing, clickbait).\n"
        "- List risk factors (unverified statistics, anonymous sources, sensationalism).\n"
        "- Rate evidence strength: 'Strong', 'Medium', or 'Weak'.\n"
        "- Write a 1-3 sentence reasoning summary explaining your credibility assessment "
        "(reasoningSummary). This MUST NOT be empty.\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "claims": [\n'
        '    {"text": "claim text", "temporal_signal": "today or null", '
        '"explicit_date": "2026-03-08 or null"}\n'
        "  ],\n"
        '  "primaryClaim": "the single most important factual claim from the text",\n'
        '  "semanticScore": 65,\n'
        '  "confidenceScore": 0.7,\n'
        '  "manipulationIndicators": ["indicator1"],\n'
        '  "riskFactors": ["factor1"],\n'
        '  "evidenceStrength": "Medium",\n'
        '  "reasoningSummary": "Brief explanation of the credibility assessment."\n'
        "}\n\n"
        "IMPORTANT: Every field above is REQUIRED. Do not omit any field. "
        "primaryClaim and reasoningSummary must always be non-empty strings."
    )

    messages = [
        {"role": "system", "content": combined_prompt},
        {"role": "user", "content": f"Analyze this content:\n\n{sanitized_text}"},
    ]

    client = get_azure_client()

    def _sdk_call() -> str:
        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    try:
        content = await asyncio.to_thread(_sdk_call)
        result = json.loads(content.strip())
        
        raw_claims = result.get("claims", [])
        structured_claims = []
        for c in raw_claims:
            if isinstance(c, str):
                structured_claims.append({"text": c, "temporal_signal": None, "explicit_date": None})
            elif isinstance(c, dict):
                structured_claims.append({
                    "text": c.get("text", str(c)),
                    "temporal_signal": c.get("temporal_signal"),
                    "explicit_date": c.get("explicit_date")
                })
        
        analysis = {
            "semanticScore": result.get("semanticScore", 50),
            "confidenceScore": result.get("confidenceScore", 0.5),
            "primaryClaim": result.get("primaryClaim", ""),
            "manipulationIndicators": result.get("manipulationIndicators", []),
            "riskFactors": result.get("riskFactors", []),
            "evidenceStrength": result.get("evidenceStrength", "Medium"),
            "reasoningSummary": result.get("reasoningSummary", result.get("explanation", ""))
        }
        
        return {
            "claims": structured_claims,
            "analysis": analysis,
            "semanticScore": analysis["semanticScore"]
        }
    except Exception as e:
        logger.error(f"Full text analysis failed: {e}")
        # Fallback to just claims if full analysis fails
        claims = await extract_claims_from_text(text)
        return {
            "claims": claims,
            "analysis": {"reasoningSummary": "Could not perform detailed semantic analysis."},
            "semanticScore": 50
        }

async def extract_claims_from_image(image_bytes: bytes) -> list:
    """Extract verifiable factual claims from an image."""
    mime_type = detect_image_mime_type(image_bytes)
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    messages = [
        {"role": "system", "content": IMAGE_CLAIM_EXTRACTION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract claims from this image."},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]
        }
    ]

    client = get_azure_client()

    def _sdk_call() -> str:
        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    try:
        content = await asyncio.to_thread(_sdk_call)
        result = safe_parse_json(content)
        raw_claims = result.get("claims", [])
        
        structured_claims = []
        for c in raw_claims:
            if isinstance(c, str):
                structured_claims.append({
                    "text": c,
                    "temporal_signal": None,
                    "explicit_date": None
                })
            elif isinstance(c, dict):
                structured_claims.append({
                    "text": c.get("text", str(c)),
                    "temporal_signal": c.get("temporal_signal"),
                    "explicit_date": c.get("explicit_date")
                })
        return structured_claims
    except Exception as e:
        logger.error(f"Image claim extraction failed: {e}")
        return []

def _build_image_forensics_prompt() -> str:
    """Build the image forensics system prompt with today's UTC date injected dynamically."""
    current_date = datetime.now(timezone.utc).strftime("%B %d, %Y")
    return f"""You are an expert AI image analyst, forensic investigator, and misinformation detection specialist.

Your task is to analyze an image across TWO distinct dimensions and return a scored analysis.

========================
CURRENT CONTEXT
========================
CURRENT_DATE: {current_date}
Standard date format: DD.MM.YYYY

========================
TWO SCORING DIMENSIONS
========================

DIMENSION 1 — MEDIA AUTHENTICITY SCORE (0-100)
How genuine and unmanipulated does this media appear?
  100 = Clearly authentic, consistent lighting/shadows, natural anatomy
  50  = Some ambiguity, minor inconsistencies
  0   = Heavy manipulation artifacts, clear fabrication

DIMENSION 2 — AI GENERATION PROBABILITY (0-100)
How likely is this image AI-generated?
  0  = Clearly a genuine photograph
  100 = Almost certainly AI-generated

========================
OUTPUT FORMAT
========================
Extract all visible timestamps or dates appearing in the image.

Examples include:
- news broadcast timestamps
- screenshot timestamps
- dates in charts
- camera timestamps
- social media timestamps

Return ONLY valid JSON in this exact format:
{{
  "visible_dates": ["08.03.2026", "2021"],
  "visual_description": "<strictly factual description of visible content>",
  "media_authenticity_score": <int 0-100>,
  "media_explanation": "<2-3 sentences explaining the media authenticity score>",
  "ai_generated_probability": <int 0-100>,
  "ai_reasoning": "<2-3 sentences explaining the AI probability score>",
  "verification_status": "VERIFIABLE",
  "confidence": <float 0.0-1.0>
}}
"""

async def analyze_image_forensics(image_bytes: bytes, accompanying_text: str = "") -> dict:
    """Analyze image forensics separating from claim extraction."""
    mime_type = detect_image_mime_type(image_bytes)
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    client = get_azure_client()
    # Build prompt fresh on each call so CURRENT_DATE is always today's UTC date
    forensics_prompt = _build_image_forensics_prompt()

    messages = [
        {"role": "system", "content": forensics_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Analyze this image across all scoring dimensions. Return JSON exactly as specified."
                },
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]
        }
    ]
    
    def _sdk_call() -> str:
        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    try:
        content = await asyncio.to_thread(_sdk_call)
        result = safe_parse_json(content)
        
        media_score = int(result.get("media_authenticity_score", 50))
        ai_prob = float(result.get("ai_generated_probability", 50)) / 100.0
        
        result["verdict"] = "Reliable" if result.get("verification_status") == "VERIFIABLE" else "Questionable"
        result["riskLevel"] = "low" if media_score > 70 else "medium"
        result["credibilityScore"] = media_score
        result["explanation"] = result.get("visual_description", "")
        result["aiGeneratedProbability"] = round(ai_prob, 4)
        result["mediaExplanation"] = result.get("media_explanation", "")
        result["aiReasoning"] = result.get("ai_reasoning", "")
        result["visible_dates"] = result.get("visible_dates", [])
        
        return result
    except Exception as e:
        logger.error(f"Image forensics failed: {e}")
        return {}

async def analyze_image_with_llm(image_bytes: bytes, accompanying_text: str = "") -> dict:
    """Main image entry point (legacy wrapper to maintain route schema)."""
    if not image_bytes:
        raise ValueError("Image bytes are required")

    claims, forensics = await asyncio.gather(
        extract_claims_from_image(image_bytes),
        analyze_image_forensics(image_bytes, accompanying_text),
    )

    return {
        "claims": claims,
        "analysis": forensics,
        "imageAuthenticityScore": forensics.get("credibilityScore", 50)
    }
