"""
TrustLens Phase 1: Semantic Credibility Analysis (Phase 6 async refactor)

The Azure OpenAI Python SDK is synchronous. Rather than blocking the event loop,
both LLM functions are wrapped with asyncio.to_thread() so the SDK call runs
in a worker thread while the event loop remains available for other coroutines.

Why asyncio.to_thread() instead of a native async Azure SDK:
  The installed openai/azure-openai SDK version uses blocking httpx under the
  hood. asyncio.to_thread() is the standard pattern for running synchronous
  third-party libraries from async code — it requires zero SDK changes and
  is trivially reversible if the SDK adds native async support later.

Why CPU-only helpers (_sanitize_user_input, _parse_and_validate_semantic_response,
detect_image_mime_type) stay synchronous:
  They do no I/O. Making them async adds coroutine overhead with zero concurrency
  benefit. Pure CPU logic MUST stay synchronous.
"""

import json
import re
import base64
import asyncio
from app.config.azure import get_azure_client
from app.config.settings import Config

# --- Shared constants / helpers (unchanged) ---

SEMANTIC_SYSTEM_PROMPT = """You are a fact-checking analyst. Your role is to extract verifiable factual claims from social media content and assess how checkable and evidence-based they are.

Your task is NOT to determine absolute truth, but to determine whether a claim is specific, factual, and potentially verifiable using external sources.

== CURRENT CONTEXT ==
CURRENT_DATE: March 11, 2026
Standard Date Format: DD.MM.YYYY

== CLAIM EXTRACTION RULES (STRICT) ==
A valid claim MUST be a complete, stand-alone factual statement containing:

  • A clear subject: person, organisation, government, event, or object
  • A specific action or event involving that subject
  • Sufficient context to understand the event

Do NOT extract:
  • Opinions, emotional reactions, or speculation
  • Calls to action
  • Sentence fragments
  • Incomplete phrases

Examples of fragments to reject:
  "For generation plus,"
  "In a week,"
  "The situation is terrible"

== DATE & TEMPORAL RULES ==

1. Compare any explicit date with CURRENT_DATE (March 11, 2026).

2. If the claim explicitly refers to a date AFTER the current date,
   mark it as a future event and note that it is not yet verifiable.

3. If a claim contains NO date, do NOT penalize it.
   Many social media posts refer to ongoing or recent events without specifying a date.

4. Absence of a date does NOT reduce credibility or verifiability.
   Treat such claims as temporally ambiguous but still potentially verifiable.

5. Interpret ambiguous numeric dates (e.g., 08.03) as DD.MM unless context clearly indicates another format.

== CLAIM VALIDITY EXAMPLES ==

VALID:
✓ "Israeli President Isaac Herzog expressed hope for Iranians to rise against their regime."
✓ "The unemployment rate rose to 8.4% in July 2025."
✓ "WHO declared a global health emergency on January 30, 2020."

INVALID:
✗ "For generation plus," ← incomplete fragment
✗ "The situation is terrible." ← opinion
✗ "Something will happen soon." ← vague claim
✗ "It will happen on March 20, 2026." ← future event

== VERIFIABILITY PRINCIPLES ==

When assessing a claim:

• Focus on whether the claim describes a concrete event or statement.
• Do NOT reduce credibility because the post lacks citations.
• Social media posts rarely include sources; this does not make a claim weaker.
• Evaluate whether the claim could theoretically be verified using credible news, official statements, or public records.

== OUTPUT STRUCTURE ==

primaryClaim:
The single most important factual claim that should be verified first.

keyClaims:
Up to 5 distinct factual claims extracted from the text.

Each claim must:
• be complete
• contain a subject and action
• be independently verifiable.

== IMPORTANT RULES ==

• Never invent missing details.
• Never infer events not present in the text.
• Never add context that is not explicitly stated.
• Extract only what is written.

Return ONLY valid JSON."""
SEMANTIC_USER_PROMPT_TEMPLATE = """Extract factual claims and assess verifiability of the following post. Return a JSON object with these exact keys. Output nothing else.

Required JSON schema:
{
  "semanticScore": <number 0-100: 100 = clear verifiable claims with strong evidence/sources; 50 = mixed or unclear claims; 0 = no checkable claims>,
  "confidenceScore": <number 0-1, your confidence in this assessment>,
  "primaryClaim": "<THE single most important factual claim to fact-check — must have a subject + action, concrete and checkable>",
  "keyClaims": ["<up to 5 specific verifiable claims — each must have a subject + action; omit vague or emotional statements>"],
  "manipulationIndicators": ["<string>", ...],
  "riskFactors": ["<string>", ...],
  "evidenceStrength": "<one of: Weak | Moderate | Strong>",
  "reasoningSummary": "<short paragraph: what factual claims were found, how verifiable they are — NOT sentiment or tone>"
}

keyClaims rules:
  - Maximum 5 claims
  - Each claim must have a clear subject (person/org/government) AND a specific action or event
  - If a statement has no subject or no action, do NOT include it
  - Omit opinions, emotional reactions, and vague sentences entirely

primaryClaim: The one claim a fact-checker should verify first (the main statistic, event, or attribution).
evidence Strength: Weak = no sources; Moderate = some attribution; Strong = verifiable sources or named authorities.
semanticScore: Base on VERIFIABILITY and EVIDENCE only — NOT on emotional language or urgency.

---POST TO ANALYZE---
{text}
---END POST---"""

VALID_EVIDENCE_STRENGTH = frozenset({"Weak", "Moderate", "Strong"})
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


def _extract_json(raw: str) -> dict:
    """
    Robust multi-strategy JSON extractor for LLM responses.

    Azure OpenAI sometimes returns:
      - Clean JSON object  ← happy path
      - JSON wrapped in ```json ... ``` fences (with optional leading newline)
      - A JSON fragment starting with \n  "key" (missing outer braces)
      - JSON embedded within explanatory text

    Strategy order:
      1. Strip markdown fences (tolerant of leading whitespace) → try json.loads
      2. Try json.loads on the raw content directly
      3. Extract first {...} block from the content → try json.loads
      4. Wrap content in braces if it looks like bare key-value pairs → try json.loads
    """
    content = raw.strip()

    # Strategy 1: strip markdown code fences (handle leading newlines before ```)
    fence_stripped = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    fence_stripped = re.sub(r"\s*```\s*$", "", fence_stripped, flags=re.MULTILINE).strip()
    try:
        result = json.loads(fence_stripped)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: try the raw content directly
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: extract first {...} block (handles JSON embedded in text)
    brace_match = re.search(r'\{.*\}', content, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: content is bare key-value pairs (missing outer braces)
    # e.g. Azure returned: \n  "semanticScore": 75,\n  "confidenceScore": 0.8, ...
    if '"semanticScore"' in content or '"primaryClaim"' in content:
        try:
            wrapped = "{" + content.strip().rstrip(",") + "}"
            result = json.loads(wrapped)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(f"Could not extract JSON from LLM response. Raw (first 200 chars): {raw[:200]!r}")


def _parse_and_validate_semantic_response(raw: str) -> dict:
    """Parse raw LLM response and validate against semantic schema."""
    parsed = _extract_json(raw)

    required_keys = [
        "semanticScore", "confidenceScore", "primaryClaim",
        "manipulationIndicators", "riskFactors", "evidenceStrength", "reasoningSummary",
    ]
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise ValueError(f"Invalid response: missing keys: {missing}")

    for key in ("manipulationIndicators", "riskFactors", "keyClaims"):
        val = parsed.get(key)
        if not isinstance(val, list):
            parsed[key] = []
        else:
            parsed[key] = [str(x) for x in val if x][:5]  # cap keyClaims at 5

    try:
        score = float(parsed.get("semanticScore", 50))
    except (TypeError, ValueError):
        score = 50.0
    parsed["semanticScore"] = max(0, min(100, round(score)))

    try:
        conf = float(parsed.get("confidenceScore", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    parsed["confidenceScore"] = max(0.0, min(1.0, round(conf, 2)))

    ev = str(parsed.get("evidenceStrength", "Weak")).strip()
    parsed["evidenceStrength"] = ev if ev in VALID_EVIDENCE_STRENGTH else "Weak"
    parsed["primaryClaim"] = str(parsed.get("primaryClaim", ""))[:500]
    parsed["reasoningSummary"] = str(parsed.get("reasoningSummary", ""))[:1000]
    return parsed


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


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------

async def analyze_text_with_llm(text: str) -> dict:
    """
    Async: Analyze social media text for credibility risk using Azure OpenAI.

    The synchronous SDK call runs in asyncio.to_thread() so it does not block
    the event loop. The rest of the function (input validation, response parsing)
    is CPU-only and runs on the event loop thread directly.
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        raise ValueError("Text input is required")

    sanitized_text = _sanitize_user_input(text)
    if not sanitized_text:
        raise ValueError("Text input is required")

    try:
        client = get_azure_client()
        user_content = SEMANTIC_USER_PROMPT_TEMPLATE.replace("{text}", sanitized_text)

        messages = [
            {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        # The Azure SDK call is synchronous — run it in a thread pool
        # so the event loop is not blocked while waiting for the network.
        def _sdk_call() -> str:
            response = client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        raw_content = await asyncio.to_thread(_sdk_call)

        if not raw_content:
            raise ValueError("No response content from Azure OpenAI")

        print(f"🔬 Raw Azure text response (first 300 chars): {raw_content[:300]!r}")
        analysis = _parse_and_validate_semantic_response(raw_content)
        return {
            "claims": analysis.get("keyClaims", []),
            "analysis": analysis,
            "semanticScore": analysis.get("semanticScore", 50)
        }

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    except ValueError:
        raise
    except Exception as e:
        print(f"Azure OpenAI text analysis failed: {e}")
        raise ValueError(f"LLM text analysis failed: {e}")


async def analyze_image_with_llm(image_bytes: bytes, accompanying_text: str = "") -> dict:
    """
    Async: Analyze image for misinformation risk using Azure OpenAI vision.
    SDK call wrapped in asyncio.to_thread() — same pattern as text analysis.
    """
    if not image_bytes:
        raise ValueError("Image bytes are required")

    try:
        mime_type = detect_image_mime_type(image_bytes)
        print(f"🔍 Detected image MIME type: {mime_type}")
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        client = get_azure_client()

        messages = [
            {
                "role": "system",
                "content": (
                    """You are an expert AI image analyst, forensic investigator, and misinformation detection specialist.

Your task is to analyze an image and extract factual information WITHOUT guessing, inventing context, or inferring events that are not visually supported.

Your analysis must strictly follow these rules.

========================
GENERAL RULES
========================

1. Only describe what is directly visible in the image.
2. Do NOT infer causes, motivations, or unseen events.
3. Do NOT assume political events, wars, protests, or attacks unless clearly visible.
4. If something cannot be confirmed visually, mark it as "uncertain".
5. Never invent text, locations, people, or events that are not present in the image.
6. If the image lacks enough information to verify a claim, return "UNVERIFIED".

========================
CURRENT CONTEXT
========================

CURRENT_DATE: March 11, 2026  
Standard date format reference: DD.MM.YYYY

This date is provided only for interpreting visible dates in the image.

========================
DATE HANDLING RULES
========================

1. Extract dates ONLY if they are explicitly visible in the image text.
2. Do NOT assume a date if one is not visible.
3. If a date appears, interpret it relative to CURRENT_DATE.
4. If the date is in the future relative to CURRENT_DATE, mark the claim as "future_event_unverifiable".
5. If the image contains no visible date, do NOT penalize the credibility or verification potential.
6. Many images (memes, screenshots, photos) naturally lack dates - this is normal.
7. If numbers appear in the image but are ambiguous (e.g., 08.03, 2026, 11/3), only interpret them as dates if the surrounding context clearly indicates a date.

========================
ANALYSIS LAYERS
========================

1. FULL TEXT EXTRACTION
- Extract ALL visible text exactly as it appears.
- Include small print, captions, overlays, labels, and background text.
- Do NOT correct spelling or rewrite text.
- If text is partially readable, mark it as "partial_text".

2. VISUAL CONTENT DESCRIPTION
Provide a strictly factual description of what is visible in the image.

Focus on:
- people
- objects
- environment
- observable actions

Important:
Only describe visible details.
Do NOT interpret meaning, intent, or narrative.

Example correct description:
"A group of people standing in a street holding signs."

Example hallucination (forbidden):
"Protesters demonstrating against government corruption."

3. VISUAL CLAIM EXTRACTION
From the visible elements, extract factual claims that could potentially be verified externally.

Each claim must contain:
- subject
- action or state
- object or context

Example:
"Several people are holding signs in a public street."

If no clear factual claims exist, return an empty claim list.

4. AI-GENERATION & MANIPULATION FORENSICS
Analyze the image for possible AI generation or manipulation artifacts.

Look for:
- distorted anatomy (hands, fingers, faces)
- inconsistent lighting or shadows
- unnatural textures
- repeating patterns
- abnormal reflections

If no artifacts are visible, return:
"ai_artifacts_detected": false

Do NOT claim AI generation unless clear artifacts are visible.

5. VERIFICATION POSSIBILITY
Assess whether the image content could potentially be verified using external sources.

Possible values:
- "VERIFIABLE"
- "PARTIALLY_VERIFIABLE"
- "UNVERIFIABLE"

Important:
Do NOT determine whether the claim is true or false.
Only evaluate whether the image provides verifiable information.

========================
OUTPUT FORMAT
========================

Return ONLY valid JSON in this exact format:
{
  "extracted_text": [],
  "visible_dates": [],
  "visual_description": "",
  "claims": [],
  "manipulation_analysis": {
      "ai_artifacts_detected": false,
      "artifact_details": ""
  },
  "verification_status": "",
  "confidence": 0.0
}"""
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Perform a rigorous multi-layered analysis of this image.\n\n"
                            "Return JSON in this exact format:\n"
                            "{\n"
                            '  "extracted_text": string[],\n'
                            '  "visible_dates": string[],\n'
                            '  "visual_description": string,\n'
                            '  "claims": string[],\n'
                            '  "manipulation_analysis": {\n'
                            '      "ai_artifacts_detected": boolean,\n'
                            '      "artifact_details": string\n'
                            '  },\n'
                            '  "verification_status": "VERIFIABLE" | "PARTIALLY_VERIFIABLE" | "UNVERIFIABLE",\n'
                            '  "confidence": number\n'
                            "}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            },
        ]

        if accompanying_text:
            user_msg = messages[1]
            content_list = user_msg.get("content")
            if isinstance(content_list, list) and len(content_list) > 0:
                first_item = content_list[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    first_item["text"] += (
                        f"\n\nThe user shared this image alongside the following text: '{accompanying_text}'.\n"
                        f"CROSS-REFERENCE VERIFICATION: Cross-reference the visual evidence in the image against this text.\n"
                        f"Determine if the image genuinely supports the user's text, or if the text drastically misrepresents the image context.\n"
                        f"Include your findings within the 'veracityCheck', 'conveyedMessage', and 'explanation' fields."
                    )

        def _sdk_call() -> str:
            response = client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        content = await asyncio.to_thread(_sdk_call)

        if not content:
            raise ValueError("No response content from Azure OpenAI")

        result = json.loads(content.strip())
        required_keys = ["visual_description", "claims", "verification_status", "confidence"]
        if not all(k in result for k in required_keys):
            raise ValueError(f"Invalid response format: missing {required_keys}")
            
        # Map new schema to existing application model
        verification_map = {
            "VERIFIABLE": "Reliable",
            "PARTIALLY_VERIFIABLE": "Questionable",
            "UNVERIFIABLE": "High Risk"
        }
        
        mapped_verdict = verification_map.get(result.get("verification_status"), "Questionable")
        risk_level = "low" if mapped_verdict == "Reliable" else ("medium" if mapped_verdict == "Questionable" else "high")
        
        # Ensure compatibility with existing fields
        result["verdict"] = mapped_verdict
        result["riskLevel"] = risk_level
        result["credibilityScore"] = int(result.get("confidence", 0.5) * 100)
        result["explanation"] = result.get("visual_description", "")
        
        return {
            "claims": result.get("claims", []),
            "analysis": result,
            "imageAuthenticityScore": result["credibilityScore"]
        }

    except Exception as e:
        print(f"Azure OpenAI image analysis failed: {e}")
        raise ValueError(f"LLM image analysis failed: {e}")
