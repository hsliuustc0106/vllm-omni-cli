"""Tests for CLI commands."""

import pytest
from typer.testing import CliRunner

from vllm_omni_cli.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "vllm-omni-cli" in result.output


def test_list_agents():
    result = runner.invoke(app, ["list-items", "agents"])
    assert result.exit_code == 0
    for name in ["architect", "coder", "optimizer", "reviewer"]:
        assert name in result.output


def test_list_tools():
    result = runner.invoke(app, ["list-items", "tools"])
    assert result.exit_code == 0
    for name in ["github", "shell", "vllm"]:
        assert name in result.output


def test_list_skills():
    result = runner.invoke(app, ["list-items", "skills"])
    assert result.exit_code == 0  # just shouldn't crash


def test_list_unknown_kind():
    result = runner.invoke(app, ["list-items", "bad"])
    assert result.exit_code != 0


def test_run_missing_task():
    # 'run' requires a task argument; invoking without should error
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_config_init(tmp_path, monkeypatch):
    import vllm_omni_cli.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(cfg_mod, "_ensure_config_dir", lambda: tmp_path.mkdir(parents=True, exist_ok=True))
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0
    assert "initialized" in result.output.lower()
