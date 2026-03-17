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
    """Backward compatibility wrapper around claim extraction."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        raise ValueError("Text input is required")
        
    claims = await extract_claims_from_text(text)
    
    return {
        "claims": claims,
        "analysis": {},
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
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    try:
        content = await asyncio.to_thread(_sdk_call)
        result = json.loads(content.strip())
        
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
        
    claims = await extract_claims_from_image(image_bytes)
    forensics = await analyze_image_forensics(image_bytes, accompanying_text)
    
    return {
        "claims": claims,
        "analysis": forensics,
        "imageAuthenticityScore": forensics.get("credibilityScore", 50)
    }
