"""Configuration management (~/.vo/config.toml)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import toml


CONFIG_DIR = Path.home() / ".vo"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG = {
    "llm": {"model": "glm-5.1", "api_key": "", "base_url": "https://open.bigmodel.cn/api/paas/v4", "default_model": "standard"},
    "tools": {"github_token": ""},
    "skills": {"paths": []},
    "defaults": {"agents": ["architect", "coder", "reviewer"], "human_in_the_loop": False},
}


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# Environment variable overrides (take precedence over config file)
_ENV_MAP = {
    ("llm", "base_url"): "VLLM_OMNI_AGENTS_BASE_URL",
    ("llm", "api_key"): "VLLM_OMNI_AGENTS_API_KEY",
    ("llm", "model"): "VLLM_OMNI_AGENTS_MODEL_NAME",
}


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config (env > config file)."""
    for (section, field), env_var in _ENV_MAP.items():
        value = os.environ.get(env_var)
        if value is not None:
            if section not in cfg:
                cfg[section] = {}
            cfg[section][field] = value
    return cfg


def load_config() -> dict[str, Any]:
    """Load config from disk, merging with defaults and env overrides."""
    _ensure_config_dir()
    if CONFIG_FILE.exists():
        data = toml.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        merged = {**DEFAULT_CONFIG}
        for section in merged:
            if section in data:
                merged[section] = {**merged[section], **data[section]}
        # Add any new sections from disk
        for section in data:
            if section not in merged:
                merged[section] = data[section]
        return _apply_env_overrides(merged)
    return _apply_env_overrides(dict(DEFAULT_CONFIG))


def save_config(cfg: dict[str, Any]) -> None:
    _ensure_config_dir()
    CONFIG_FILE.write_text(toml.dumps(cfg), encoding="utf-8")


def config_init() -> dict[str, Any]:
    """Initialize config with defaults. Returns the config."""
    _ensure_config_dir()
    cfg = load_config()
    save_config(cfg)
    return cfg


def config_set(key: str, value: str) -> None:
    """Set a config value like 'llm.model' = 'claude-3-opus'."""
    cfg = load_config()
    parts = key.split(".")
    if len(parts) == 2:
        section, field = parts
        if section not in cfg:
            cfg[section] = {}
        # Try to parse as native type
        if value.lower() in ("true", "false"):
            cfg[section][field] = value.lower() == "true"
        else:
            try:
                cfg[section][field] = int(value)
            except ValueError:
                cfg[section][field] = value
    else:
        cfg[key] = value
    save_config(cfg)


def config_get(key: str) -> str:
    """Get a config value."""
    cfg = load_config()
    parts = key.split(".")
    if len(parts) == 2:
        return str(cfg.get(parts[0], {}).get(parts[1], ""))
    return str(cfg.get(key, ""))


def config_list() -> dict[str, Any]:
    return load_config()
