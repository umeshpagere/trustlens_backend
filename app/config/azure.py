import os
from openai import AsyncAzureOpenAI, AzureOpenAI
from app.config.settings import Config

def get_azure_client():
    """Get Azure OpenAI client instance."""
    return AzureOpenAI(
        api_key=Config.AZURE_OPENAI_API_KEY,
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_version=Config.AZURE_OPENAI_API_VERSION
    )

def get_async_azure_client():
    """Get async Azure OpenAI client instance."""
    return AsyncAzureOpenAI(
        api_key=Config.AZURE_OPENAI_API_KEY,
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_version=Config.AZURE_OPENAI_API_VERSION
    )
