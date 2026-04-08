"""Tool for resolving ambiguous model aliases into known model families."""

from __future__ import annotations

from typing import Any

from ..core.tool import BaseTool
from ..model_catalog import resolve_model_alias


class ModelResolverTool(BaseTool):
    """Resolve ambiguous model names used in prompts."""

    name = "model_resolver"
    description = "Resolve an ambiguous model name to a known family and suggested checkpoints."
    category = "domain"
    scopes = ["architect", "coder", "optimizer", "reviewer"]
    parameters = {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "User-provided model name or alias to resolve.",
            },
        },
        "required": ["model_name"],
    }

    async def execute(self, **kwargs: Any) -> str:
        model_name = kwargs.get("model_name", "")
        entry = resolve_model_alias(model_name)
        if entry is None:
            return (
                f"No catalog entry found for '{model_name}'. "
                "Treat it as user-defined or ask for the exact checkpoint."
            )
        suggestions = ", ".join(entry.suggestions) if entry.suggestions else "none"
        return (
            f"alias={entry.alias}\n"
            f"status={entry.status}\n"
            f"family={entry.canonical_family}\n"
            f"note={entry.note}\n"
            f"suggestions={suggestions}"
        )
