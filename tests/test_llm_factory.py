"""Tests for LLMFactory."""

import pytest

from vllm_omni_cli.core.llm_factory import LLMFactory, AGENT_LEVEL_MAP, DEFAULT_AGENT_LEVEL, DEFAULT_MODEL_LEVELS


def test_factory_create_with_defaults():
    factory = LLMFactory()
    backend = factory.create()
    assert backend.model == "glm-5.1"
    assert backend.temperature == 0.2
    assert backend.max_tokens == 8192


def test_factory_create_with_model_level():
    config = {
        "default_model": "standard",
        "model": "base-model",
        "models": {
            "complex": {"model": "complex-model", "temperature": 0.1, "max_tokens": 16384},
            "standard": {"model": "standard-model", "temperature": 0.2, "max_tokens": 8192},
            "fast": {"model": "fast-model", "temperature": 0.3, "max_tokens": 4096},
        },
    }
    factory = LLMFactory(config=config)

    complex_backend = factory.create(model_level="complex")
    assert complex_backend.model == "complex-model"
    assert complex_backend.temperature == 0.1
    assert complex_backend.max_tokens == 16384

    fast_backend = factory.create(model_level="fast")
    assert fast_backend.model == "fast-model"
    assert fast_backend.temperature == 0.3
    assert fast_backend.max_tokens == 4096


def test_factory_create_direct_override():
    config = {
        "models": {
            "standard": {"model": "standard-model", "temperature": 0.2},
        },
    }
    factory = LLMFactory(config=config)
    backend = factory.create(model="override-model", api_key="sk-test", base_url="http://custom:8000")
    assert backend.model == "override-model"
    assert backend.api_key == "sk-test"
    assert backend.base_url == "http://custom:8000"


def test_factory_list_levels():
    factory = LLMFactory()
    levels = factory.list_levels()
    assert "complex" in levels
    assert "standard" in levels
    assert "fast" in levels


def test_factory_create_for_agent():
    factory = LLMFactory(config={
        "models": {
            "complex": {"model": "deepseek-reasoner", "temperature": 0.1},
            "standard": {"model": "deepseek-chat", "temperature": 0.2},
        },
    })
    arch_llm = factory.create_for_agent("architect")
    assert arch_llm.model == "deepseek-reasoner"
    coder_llm = factory.create_for_agent("coder")
    assert coder_llm.model == "deepseek-chat"
    unknown_llm = factory.create_for_agent("unknown_agent")
    assert unknown_llm.model == "deepseek-chat"


def test_factory_extra_body():
    config = {
        "models": {
            "complex": {
                "model": "deepseek-reasoner",
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        },
    }
    factory = LLMFactory(config=config)
    backend = factory.create(model_level="complex")
    assert backend.extra_body == {"thinking": {"type": "enabled"}}


def test_factory_token_accumulation():
    factory = LLMFactory()
    backend = factory.create()
    assert backend.total_prompt_tokens == 0
    assert backend.total_completion_tokens == 0
    assert backend.total_tokens == 0


def test_agent_level_map():
    assert AGENT_LEVEL_MAP["architect"] == "complex"
    assert AGENT_LEVEL_MAP["optimizer"] == "complex"
    assert AGENT_LEVEL_MAP["coder"] == "standard"
    assert AGENT_LEVEL_MAP["reviewer"] == "standard"
    assert DEFAULT_AGENT_LEVEL == "standard"


def test_factory_uses_default_model_from_config():
    config = {"default_model": "fast", "models": {"fast": {"model": "fast-model"}}}
    factory = LLMFactory(config=config)
    backend = factory.create()  # no model_level specified
    assert backend.model == "fast-model"


def test_factory_level_config_falls_back_to_global():
    config = {"model": "global-model", "models": {"standard": {"temperature": 0.5}}}
    factory = LLMFactory(config=config)
    backend = factory.create(model_level="standard")
    assert backend.model == "global-model"
    assert backend.temperature == 0.5
