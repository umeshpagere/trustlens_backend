import logging

logger = logging.getLogger(__name__)

import json
from openai import AzureOpenAI
from app.config.settings import Config

def decompose_claim(claim: str) -> list:
    if not Config.AZURE_OPENAI_API_KEY or not Config.AZURE_OPENAI_ENDPOINT:
        logger.error("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT is not set.")
        return []
        
    client = AzureOpenAI(
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
    )
    prompt = """\
You are an EVENT-LEVEL CLAIM DECOMPOSITION SYSTEM used in a fact-verification pipeline.

Your task is to convert raw claim strings into structured, event-level factual claims that can be verified using news, fact-check, and knowledge databases.

The goal is to extract ONLY the core real-world events that journalists or fact-checkers would verify.

------------------------------------------------

STAGE 1 — EVENT VALIDATION

Accept a claim ONLY if it describes a CLEAR, VERIFIABLE EVENT.

A valid event must include:

• a subject (person, organization, government, object)
• an action (what happened)
• a concrete outcome or object (optional but preferred)

The claim must describe something that could appear as a NEWS HEADLINE.

REJECT the following types of sentences:

• topic summaries
• vague descriptions
• contextual information
• background explanations
• investment counts
• statements about companies or spokespersons
• commentary about the event

Examples:

BAD (REJECT)
"Iran and USA war today"
"Flood situation getting worse"
"This is Amitabh Bachchan's third investment in Ayodhya"
"The realty company said the deal was completed"

GOOD (ACCEPT)
"US fighter jet shot down an Iranian fighter jet over the Gulf"
"Amitabh Bachchan purchased land in Ayodhya"
"Government declared emergency after flooding in Mumbai"

If a sentence only provides context about an event rather than describing the event itself, REJECT it.

------------------------------------------------

STAGE 2 — CLAIM STRUCTURING

For each accepted event extract:

claim:
  A clean factual sentence describing the event.

subject:
  The main actor (person, organization, government).

action:
  The event verb (short phrase).

object:
  The target or outcome of the action.
  Use an empty string if none exists.

context:
  A short 3–5 word phrase describing the background topic.

Example:

Input:
"Amitabh Bachchan purchased a 2.67-acre plot in Ayodhya for \u20b935 crore."

Output fields:

subject: "Amitabh Bachchan"
action: "purchased"
object: "2.67-acre plot in Ayodhya"
context: "Ayodhya real estate investment"

------------------------------------------------

STAGE 3 — QUERY NORMALIZATION

Generate 2–3 short search queries optimized for retrieving evidence.

Rules:

• 3–7 words only
• MUST contain the main subject
• Focus on the event, not background context
• Avoid long sentences
• Each query should be phrased differently

Example queries:

"Amitabh Bachchan Ayodhya land purchase"
"Amitabh Bachchan bought land Ayodhya"
"Amitabh Bachchan Ayodhya property investment"

These queries will be used with:

• Google Fact Check API
• Wikipedia search
• News API

------------------------------------------------

IMPORTANT RULES

1. Prioritize the MAIN EVENT claim if multiple claims describe the same story.

2. Never output contextual claims like:
   "This is his third investment."

3. If a claim describes an event reported by news media, it should be included.

4. Prefer claims that follow a news headline structure.

------------------------------------------------

OUTPUT FORMAT

Return ONLY a JSON array with no additional text.

[
  {
    "claim": "<event-level factual sentence>",
    "subject": "<actor/entity>",
    "action": "<verb/event>",
    "object": "<target/outcome or empty>",
    "context": "<3-5 word background>",
    "normalizedQueries": ["<query 1>", "<query 2>", "<query 3>"]
  }
]

------------------------------------------------

OUTPUT LIMITS

• Maximum 5 structured claims
• If no valid event claims exist, return []

------------------------------------------------

Input Claim: {claim}
"""
    try:
        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt.replace("{claim}", claim)}],
        )
        content = response.choices[0].message.content
        # sometimes LLMs return json wrapped in ```json ... ``` blocks
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        return json.loads(content.strip())
    except Exception as e:
        logger.error(f"Error decomposing claim: {e}")
        return []
