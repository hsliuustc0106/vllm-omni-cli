"""Shared request preparation for run, quick, and chat flows."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..agents import BUILTIN_AGENTS
from ..config import config_get
from ..model_catalog import ModelAliasEntry, load_model_alias_entries
from .skill import SkillRegistry

ROUTING_RULES_FILE = Path(__file__).parent.parent / "data" / "routing_rules.json"


class AgentRequest(BaseModel):
    request_id: str
    task_text: str
    mode: Literal["orchestrated", "quick", "chat"]
    target_agents: list[str]
    explicit_model: str | None = None
    resolved_model_aliases: list[str] = Field(default_factory=list)
    resolved_model_family: str | None = None
    task_categories: list[str] = Field(default_factory=list)
    hardware_hints: list[str] = Field(default_factory=list)
    manual_skill_refs: list[str] = Field(default_factory=list)
    auto_skill_refs: list[str] = Field(default_factory=list)
    merged_skill_refs: list[str] = Field(default_factory=list)
    tool_scope_agents: list[str] = Field(default_factory=list)
    debug: bool = False
    human_in_the_loop: bool = False
    routing_notes: list[str] = Field(default_factory=list)


class ChatSessionState(BaseModel):
    resolved_model_family: str | None = None
    resolved_model_aliases: list[str] = Field(default_factory=list)
    selected_skills: list[str] = Field(default_factory=list)
    routing_notes: list[str] = Field(default_factory=list)


def load_routing_rules() -> dict:
    return json.loads(ROUTING_RULES_FILE.read_text(encoding="utf-8"))


def load_installed_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    paths_str = config_get("skills.paths")
    if not paths_str:
        return registry

    import ast

    paths = ast.literal_eval(paths_str)
    for path_str in paths:
        path = Path(path_str)
        if path.exists():
            registry.load_from_directory(path)
    return registry


def prepare_agent_request(
    task_text: str,
    mode: Literal["orchestrated", "quick", "chat"],
    *,
    explicit_model: str | None = None,
    manual_skill_refs: list[str] | None = None,
    target_agents: list[str] | None = None,
    debug: bool = False,
    human_in_the_loop: bool = False,
    skill_registry: SkillRegistry | None = None,
    chat_session: ChatSessionState | None = None,
) -> AgentRequest:
    rules = load_routing_rules()
    normalized = task_text.lower()

    alias_entries = _resolve_alias_entries(normalized)
    resolved_aliases = [entry.alias for entry in alias_entries]
    resolved_family = alias_entries[0].canonical_family if alias_entries else None
    routing_notes: list[str] = []

    if not resolved_family and chat_session and chat_session.resolved_model_family:
        resolved_family = chat_session.resolved_model_family
        resolved_aliases = list(chat_session.resolved_model_aliases)
        routing_notes.append("Using sticky chat-session model family.")

    categories, hardware_hints = _classify_categories(normalized, rules, resolved_family)
    auto_skill_refs: list[str] = []
    if skill_registry is None:
        routing_notes.append("No SkillRegistry provided; skipping auto-skill selection.")
    else:
        auto_skill_refs = _select_auto_skills(resolved_family, categories, skill_registry, rules)

    manual_skill_refs = manual_skill_refs or []
    merged_skill_refs = _merge_skill_refs(manual_skill_refs, auto_skill_refs)
    final_target_agents = _select_target_agents(
        mode=mode,
        requested_agents=target_agents or [],
        resolved_family=resolved_family,
        rules=rules,
    )

    if resolved_family:
        routing_notes.append(f"Resolved family: {resolved_family}")
    if categories:
        routing_notes.append(f"Primary category: {categories[0]}")

    request = AgentRequest(
        request_id=str(uuid.uuid4()),
        task_text=task_text,
        mode=mode,
        target_agents=final_target_agents,
        explicit_model=explicit_model,
        resolved_model_aliases=resolved_aliases,
        resolved_model_family=resolved_family,
        task_categories=categories,
        hardware_hints=hardware_hints,
        manual_skill_refs=manual_skill_refs,
        auto_skill_refs=auto_skill_refs,
        merged_skill_refs=merged_skill_refs,
        tool_scope_agents=list(final_target_agents),
        debug=debug,
        human_in_the_loop=human_in_the_loop,
        routing_notes=routing_notes,
    )

    if chat_session is not None:
        chat_session.resolved_model_family = request.resolved_model_family
        chat_session.resolved_model_aliases = list(request.resolved_model_aliases)
        chat_session.selected_skills = list(request.merged_skill_refs)
        chat_session.routing_notes = list(request.routing_notes)

    return request


def render_prepared_task(request: AgentRequest) -> str:
    sections: list[str] = []
    if request.resolved_model_family:
        rules = load_routing_rules()
        family_type = rules.get("family_type_map", {}).get(request.resolved_model_family, "unknown")
        do_not_substitute = rules.get("family_forbidden_map", {}).get(request.resolved_model_family)
        source = _resolved_family_source(request.resolved_model_aliases)
        resolved_block = [
            "[Resolved Model]",
            f"request_id: {request.request_id}",
            f"alias: {', '.join(request.resolved_model_aliases) if request.resolved_model_aliases else '(none)'}",
            f"family: {request.resolved_model_family}",
            f"type: {family_type}",
            f"source: {source}",
        ]
        if do_not_substitute:
            resolved_block.append(f"do_not_substitute: {do_not_substitute}")
        sections.append("\n".join(resolved_block))

    if request.task_categories or request.merged_skill_refs or request.target_agents:
        routing_block = [
            "[Routing]",
            f"primary_category: {request.task_categories[0] if request.task_categories else 'general-architecture'}",
            f"extra_categories: {', '.join(request.task_categories[1:]) if len(request.task_categories) > 1 else '(none)'}",
            f"selected_skills: {', '.join(request.merged_skill_refs) if request.merged_skill_refs else '(none)'}",
            f"selected_agent: {request.target_agents[0] if request.target_agents else 'architect'}",
        ]
        if request.hardware_hints:
            routing_block.append(f"hardware_hints: {', '.join(request.hardware_hints)}")
        sections.append("\n".join(routing_block))

    if request.routing_notes and request.debug:
        sections.append("[Routing Notes]\n" + "\n".join(f"- {note}" for note in request.routing_notes))

    sections.append(request.task_text)
    return "\n\n".join(sections)


def _resolve_alias_entries(normalized_text: str) -> list[ModelAliasEntry]:
    matches: list[ModelAliasEntry] = []
    for entry in load_model_alias_entries():
        if entry.alias in normalized_text:
            matches.append(entry)
    return matches


def _classify_categories(
    normalized_text: str,
    rules: dict,
    resolved_family: str | None,
) -> tuple[list[str], list[str]]:
    matched_categories: set[str] = set()
    hardware_hints: set[str] = set()

    for rule in rules.get("keyword_rules", []):
        category = rule["category"]
        for keyword in rule.get("keywords", []):
            if keyword in normalized_text:
                matched_categories.add(category)
                if category == "hardware-topology":
                    hardware_hints.add(keyword)

    family_category_map = rules.get("family_category_map", {})
    if resolved_family and resolved_family in family_category_map:
        matched_categories.add(family_category_map[resolved_family])

    precedence = rules.get("precedence", [])
    ordered = [category for category in precedence if category in matched_categories]
    if not ordered:
        ordered = ["general-architecture"]
    return ordered, sorted(hardware_hints)


def _select_auto_skills(
    resolved_family: str | None,
    categories: list[str],
    skill_registry: SkillRegistry,
    rules: dict,
) -> list[str]:
    selected: list[str] = []
    family_skill_map = rules.get("family_skill_map", {})
    if resolved_family:
        for skill_name in family_skill_map.get(resolved_family, []):
            if skill_registry.exists(skill_name):
                selected.append(skill_name)

    if "hardware-topology" in categories and skill_registry.exists("vllm-omni-hardware"):
        selected.append("vllm-omni-hardware")

    return list(dict.fromkeys(selected))


def _merge_skill_refs(manual_skill_refs: list[str], auto_skill_refs: list[str]) -> list[str]:
    merged: list[str] = []
    seen: dict[str, str] = {}

    def _identity(ref: str) -> str:
        path = Path(ref)
        return path.resolve().as_posix() if path.exists() else ref

    for ref in auto_skill_refs:
        ident = _identity(ref)
        if ident not in seen:
            seen[ident] = ref
            merged.append(ref)

    for ref in manual_skill_refs:
        path = Path(ref)
        if path.exists():
            # Path-based manual refs win over existing name-based refs when identical strings differ.
            merged = [existing for existing in merged if existing != path.name]
        ident = _identity(ref)
        if ident not in seen:
            seen[ident] = ref
            merged.append(ref)
    return merged


def _select_target_agents(
    *,
    mode: Literal["orchestrated", "quick", "chat"],
    requested_agents: list[str],
    resolved_family: str | None,
    rules: dict,
) -> list[str]:
    if requested_agents:
        return requested_agents

    if mode == "orchestrated":
        return list(BUILTIN_AGENTS.keys())

    if mode in {"quick", "chat"}:
        default_agent = rules.get("family_agent_map", {}).get(resolved_family or "", "architect")
        return [default_agent]

    return ["architect"]


def _resolved_family_source(resolved_aliases: list[str]) -> str:
    alias_map = {entry.alias: entry for entry in load_model_alias_entries()}
    for alias in resolved_aliases:
        entry = alias_map.get(alias)
        if entry is not None:
            return entry.source
    return "builtin"
