"""Tests for config module."""

import os
import pytest
from pathlib import Path

from vllm_omni_cli.config import (
    DEFAULT_CONFIG,
    load_config,
    save_config,
    config_init,
    config_set,
    config_get,
    config_list,
    CONFIG_FILE,
)


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    """Redirect config to a temp directory."""
    monkeypatch.setattr("vllm_omni_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("vllm_omni_cli.config.CONFIG_FILE", tmp_path / "config.toml")
    # Ensure _ensure_config_dir uses the temp dir
    import vllm_omni_cli.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "_ensure_config_dir", lambda: tmp_path.mkdir(parents=True, exist_ok=True))
    return tmp_path


def test_config_init_creates_file(temp_config):
    cfg = config_init()
    assert (temp_config / "config.toml").exists()
    assert "llm" in cfg
    assert cfg["llm"]["model"] == "glm-5.1"


def test_config_set_get_roundtrip(temp_config):
    config_init()
    config_set("llm.model", "claude-3-opus")
    assert config_get("llm.model") == "claude-3-opus"


def test_config_list(temp_config):
    config_init()
    cfg = config_list()
    assert isinstance(cfg, dict)
    assert "llm" in cfg
    assert "tools" in cfg
    assert "defaults" in cfg


def test_env_override(monkeypatch, temp_config):
    monkeypatch.setenv("VLLM_OMNI_AGENTS_BASE_URL", "http://custom:8000")
    config_init()
    assert config_get("llm.base_url") == "http://custom:8000"


def test_config_set_bool(temp_config):
    config_init()
    config_set("defaults.human_in_the_loop", "true")
    assert config_get("defaults.human_in_the_loop") == "True"


def test_config_get_missing(temp_config):
    config_init()
    assert config_get("nonexistent.key") == ""
