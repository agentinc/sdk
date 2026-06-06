from __future__ import annotations

from ..schemas import ModelConfig
from .base import Provider


def provider_for(config: ModelConfig) -> Provider:
    """Return the correct Provider instance based on model name or base_url."""
    model = config["model"]
    base_url = config.get("base_url")

    if base_url:
        from .openai import OpenAIProvider
        return OpenAIProvider(config)

    if model.startswith("claude"):
        from .anthropic import AnthropicProvider
        return AnthropicProvider(config)

    if model.startswith("gemini"):
        from .gemini import GeminiProvider
        return GeminiProvider(config)

    # Default: OpenAI (covers gpt-*, o1-*, o3-*, etc.)
    from .openai import OpenAIProvider
    return OpenAIProvider(config)


__all__ = ["Provider", "provider_for"]
