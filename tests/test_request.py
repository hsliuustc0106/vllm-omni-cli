"""Tests for shared request preparation."""

from vllm_omni_cli.core.request import (
    ChatSessionState,
    load_routing_rules,
    prepare_agent_request,
    render_prepared_task,
)
from vllm_omni_cli.core.skill import SkillRegistry


def test_prepare_agent_request_resolves_qwen_image_family():
    request = prepare_agent_request(
        "Design the least latency deployment strategy for qwen-image on 2x L20",
        "quick",
        target_agents=[],
        skill_registry=SkillRegistry(),
    )

    assert request.resolved_model_family == "Qwen-Image family"
    assert request.task_categories[0] == "text-to-image"
    assert "hardware-topology" in request.task_categories
    assert "l20" in request.hardware_hints
    assert request.target_agents == ["architect"]


def test_prepare_agent_request_chat_uses_sticky_family():
    session = ChatSessionState(resolved_model_family="Qwen-Image family", resolved_model_aliases=["qwen-image"])
    request = prepare_agent_request(
        "now benchmark it",
        "chat",
        target_agents=["architect"],
        skill_registry=SkillRegistry(),
        chat_session=session,
    )

    assert request.resolved_model_family == "Qwen-Image family"
    assert "Using sticky chat-session model family." in request.routing_notes
    assert "performance-optimization" in request.task_categories


def test_render_prepared_task_contains_structured_blocks():
    request = prepare_agent_request(
        "Deploy qwen-image on 2x L20",
        "quick",
        target_agents=[],
        skill_registry=SkillRegistry(),
        debug=True,
    )

    rendered = render_prepared_task(request)
    assert "[Resolved Model]" in rendered
    assert "family: Qwen-Image family" in rendered
    assert "[Routing]" in rendered
    assert "selected_agent: architect" in rendered
    assert "request_id:" in rendered


def test_load_routing_rules_uses_expected_categories():
    rules = load_routing_rules()
    assert "text-to-image" in rules["categories"]
    assert rules["family_agent_map"]["Qwen-Image family"] == "architect"
