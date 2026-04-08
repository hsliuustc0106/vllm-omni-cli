"""Tests for model catalog and recipes sync helpers."""

from vllm_omni_cli.model_catalog import build_model_resolution_note, resolve_model_alias
from vllm_omni_cli.recipes_sync import normalize_alias


def test_resolve_model_alias_builtin_entry():
    entry = resolve_model_alias("qwen-image")
    assert entry is not None
    assert entry.canonical_family == "Qwen-Image family"


def test_build_model_resolution_note_mentions_alias():
    note = build_model_resolution_note("deploy qwen-image on 2x l20")
    assert "Model alias note for 'qwen-image'" in note


def test_normalize_alias():
    assert normalize_alias("Qwen3-VL") == "qwen3-vl"
    assert normalize_alias("GLM-4.5V, GLM-4.6V") == "glm-4.5v-glm-4.6v"
