import logging

logger = logging.getLogger(__name__)

import json
from openai import AzureOpenAI
from app.config.settings import Config

def extract_claims(text: str) -> dict:
    if not Config.AZURE_OPENAI_API_KEY or not Config.AZURE_OPENAI_ENDPOINT:
        logger.error("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT is not set.")
        return {"primaryClaim": "", "keyClaims": []}
        
    client = AzureOpenAI(
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
    )
    prompt = """
    Extract the primary claim and key sub-claims from the following text based on this exactly matched JSON schema:
    {
      "primaryClaim": "The main claim...",
      "keyClaims": [
        "First sub claim...",
        "Second sub claim..."
      ]
    }
    
    Text: {text}
    """
    try:
        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt.replace("{text}", text)}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error extracting claims: {e}")
        return {"primaryClaim": "", "keyClaims": []}
