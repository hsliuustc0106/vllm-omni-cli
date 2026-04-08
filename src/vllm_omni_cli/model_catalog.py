"""Local model alias catalog used for prompt normalization and agent tooling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CONFIG_DIR = Path.home() / ".vo"
USER_CATALOG_FILE = CONFIG_DIR / "model_aliases.json"
BUILTIN_CATALOG_FILE = Path(__file__).with_name("data") / "model_aliases.json"


@dataclass(frozen=True)
class ModelAliasEntry:
    alias: str
    status: str
    canonical_family: str
    note: str
    suggestions: tuple[str, ...] = ()
    source: str = "builtin"


def _load_catalog_file(path: Path, source: str) -> list[ModelAliasEntry]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: list[ModelAliasEntry] = []
    for item in raw:
        entries.append(
            ModelAliasEntry(
                alias=item["alias"],
                status=item["status"],
                canonical_family=item["canonical_family"],
                note=item["note"],
                suggestions=tuple(item.get("suggestions", [])),
                source=item.get("source", source),
            )
        )
    return entries


def load_model_alias_entries() -> tuple[ModelAliasEntry, ...]:
    """Load builtin catalog plus optional user-synced overrides."""
    merged: dict[str, ModelAliasEntry] = {}
    for entry in _load_catalog_file(BUILTIN_CATALOG_FILE, "builtin"):
        merged[entry.alias] = entry
    for entry in _load_catalog_file(USER_CATALOG_FILE, "user"):
        merged[entry.alias] = entry
    return tuple(merged.values())


def resolve_model_alias(name: str) -> ModelAliasEntry | None:
    """Resolve a user-supplied model alias to a known family entry."""
    normalized = name.strip().lower()
    for entry in load_model_alias_entries():
        if normalized == entry.alias:
            return entry
    return None


def build_model_resolution_note(text: str) -> str:
    """Generate a brief clarification note for ambiguous model references in free text."""
    normalized = text.lower()
    notes: list[str] = []
    for entry in load_model_alias_entries():
        if entry.alias in normalized:
            suggestions = ", ".join(entry.suggestions[:3]) if entry.suggestions else "none"
            notes.append(
                f"Model alias note for '{entry.alias}': status={entry.status}; "
                f"family={entry.canonical_family}; note={entry.note}; "
                f"example checkpoints={suggestions}. "
                "Treat this mapping as authoritative for this answer. "
                "Do not substitute a different similarly named family unless the user explicitly asks for it."
            )
    return "\n".join(notes)
