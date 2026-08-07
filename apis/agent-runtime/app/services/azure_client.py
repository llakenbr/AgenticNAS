from functools import lru_cache

from langchain_openai import AzureChatOpenAI

from core.settings import settings
from models.agent import ModelConfig


def get_azure_llm(model: ModelConfig) -> AzureChatOpenAI:
    """Return a cached AzureChatOpenAI client keyed on the model parameters."""
    return _build_llm(model.model_name, model.temperature, model.streaming)


@lru_cache(maxsize=16)
def _build_llm(model_name: str, temperature: float, streaming: bool) -> AzureChatOpenAI:
    """Inner cached factory — all args are hashable scalars."""
    api_key = settings.azure_openai_api_key_resolved
    endpoint = settings.azure_openai_endpoint_resolved
    if not api_key or not endpoint:
        raise RuntimeError(
            "Azure OpenAI credentials are not configured. "
            "Check AZURE_OPENAI_API_KEY_FILE and AZURE_OPENAI_API_ENDPOINT_FILE."
        )
    return AzureChatOpenAI(
        azure_deployment=model_name,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=settings.azure_openai_api_version,
        temperature=temperature,
        streaming=streaming,
    )