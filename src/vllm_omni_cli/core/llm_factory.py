"""LLM Factory - create LLMBackend instances from config."""

import logging
from .llm import LLMBackend

logger = logging.getLogger(__name__)

# Default model: GLM-5.1 (智谱旗舰模型，支持 Agentic Coding)
# API: https://open.bigmodel.cn/api/paas/v4/chat/completions
# litellm prefix: openai/ (uses OpenAI-compatible provider)
DEFAULT_MODEL = "glm-5.1"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# Default model levels
DEFAULT_MODEL_LEVELS = {
    "complex": {
        "description": "Powerful models for complex reasoning tasks (architect, optimizer)",
        "priority": 0,
        "default": {"model": "glm-5.1", "extra_body": {"thinking": {"type": "enabled"}}, "temperature": 1.0, "max_tokens": 65536},
    },
    "standard": {
        "description": "Standard models for general tasks (coder, reviewer)",
        "priority": 1,
        "default": {"model": "glm-5.1", "temperature": 0.2, "max_tokens": 16384},
    },
    "fast": {
        "description": "Fast models for simple tasks (planning, summarization)",
        "priority": 2,
        "default": {"model": "glm-5.1", "temperature": 0.3, "max_tokens": 4096},
    },
}

# Map agent roles to default model levels
AGENT_LEVEL_MAP = {
    "architect": "complex",
    "optimizer": "complex",
    "coder": "standard",
    "reviewer": "standard",
}

# Fallback level for unmapped agents
DEFAULT_AGENT_LEVEL = "standard"


class LLMFactory:
    """Factory for creating LLMBackend instances.

    Supports model levels (complex/standard/fast) that can be configured
    via config file or environment variables. Also allows direct parameter override.

    Usage:
        factory = LLMFactory(config)

        # By level
        client = factory.create(model_level="complex")

        # By agent name (uses AGENT_LEVEL_MAP)
        client = factory.create_for_agent("architect")

        # By direct params (override config)
        client = factory.create(model="deepseek-chat", api_key="sk-...", base_url="...")

        # Default
        client = factory.create()  # uses default_model from config
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}

    def create(
        self,
        model_level: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        **kwargs,
    ) -> LLMBackend:
        """Create an LLMBackend instance.

        Priority: direct params > model_level config > default config

        Returns:
            LLMBackend instance
        """
        # If no model_level specified, use default
        if model_level is None:
            model_level = self._config.get("default_model", "standard")

        # Get level-specific config
        level_config = self._config.get("models", {}).get(model_level, {})

        # Merge: direct params > level config > defaults
        final_model = model or level_config.get("model") or self._config.get("model", "glm-5.1")
        final_api_key = api_key or level_config.get("api_key") or self._config.get("api_key")
        final_base_url = base_url or level_config.get("base_url") or self._config.get("base_url")
        final_temperature = temperature if temperature is not None else level_config.get("temperature", 0.2)
        final_max_tokens = max_tokens if max_tokens is not None else level_config.get("max_tokens", 8192)
        final_extra_body = extra_body or level_config.get("extra_body")

        backend = LLMBackend(
            model=final_model,
            api_key=final_api_key,
            base_url=final_base_url,
            temperature=final_temperature,
            max_tokens=final_max_tokens,
            extra_body=final_extra_body,
            **kwargs,
        )

        logger.info(f"Created LLMBackend: level={model_level}, model={final_model}")
        return backend

    def create_for_agent(self, agent_name: str) -> LLMBackend:
        """Create LLMBackend for a specific agent using AGENT_LEVEL_MAP."""
        level = AGENT_LEVEL_MAP.get(agent_name, DEFAULT_AGENT_LEVEL)
        return self.create(model_level=level)

    def list_levels(self) -> dict:
        """List available model levels."""
        return {k: v.get("description", "") for k, v in DEFAULT_MODEL_LEVELS.items()}
