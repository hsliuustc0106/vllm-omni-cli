"""Sync model alias hints from the vllm-project/recipes repository."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from .model_catalog import CONFIG_DIR, USER_CATALOG_FILE, _load_catalog_file


RECIPES_TREE_API = "https://api.github.com/repos/vllm-project/recipes/git/trees/main?recursive=2"
IGNORED_ROOT_NAMES = {
    ".gitignore",
    ".readthedocs.yaml",
    "README.md",
    "LICENSE",
    "mkdocs.yml",
    "requirements.txt",
}


def normalize_alias(text: str) -> str:
    """Normalize recipe names into lowercase alias keys."""
    normalized = text.strip().lower()
    normalized = normalized.replace("_", "-").replace("/", "-")
    normalized = re.sub(r"[^a-z0-9.+-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized


def _github_json(url: str) -> Any:
    response = httpx.get(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "vllm-omni-cli",
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _entry_from_recipe(vendor: str, recipe_name: str) -> dict[str, Any]:
    alias = normalize_alias(recipe_name)
    family = f"{vendor} recipe family"
    note = f"Imported from vllm-project/recipes. '{recipe_name}' is a documented recipe under vendor '{vendor}'."
    suggestions = [f"{vendor}/{recipe_name}"]

    if vendor == "Qwen" and recipe_name == "Qwen-Image":
        family = "Qwen-Image family"
        note = (
            "Imported from vllm-project/recipes. 'Qwen-Image' is a text-to-image / image-editing "
            "DiT model family. It is not the same family as Qwen-VL and should not be interpreted "
            "as a vision-language chat model."
        )

    return {
        "alias": alias,
        "status": "recipe",
        "canonical_family": family,
        "note": note,
        "suggestions": suggestions,
        "source": "vllm-project/recipes",
    }


def sync_recipes_catalog(output_path: Path | None = None) -> dict[str, Any]:
    """Fetch recipe names from GitHub and write them into the user catalog."""
    output_path = output_path or USER_CATALOG_FILE
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    tree_payload = _github_json(RECIPES_TREE_API)
    tree_entries = tree_payload.get("tree", [])
    imported_entries: list[dict[str, Any]] = []
    seen_aliases: set[str] = set()

    for item in tree_entries:
        path = item.get("path", "")
        if item.get("type") != "blob" or "/" not in path:
            continue
        parts = path.split("/")
        if len(parts) != 2:
            continue
        vendor, recipe_file = parts
        if vendor in IGNORED_ROOT_NAMES:
            continue
        if not recipe_file.endswith((".md", ".ipynb", ".yaml", ".yml")):
            continue
        recipe_name = recipe_file.rsplit(".", 1)[0]
        entry = _entry_from_recipe(vendor, recipe_name)
        if entry["alias"] in seen_aliases:
            continue
        seen_aliases.add(entry["alias"])
        imported_entries.append(entry)

    existing_entries = _load_catalog_file(output_path, "user")
    preserved_entries = []
    for entry in existing_entries:
        if entry.source != "vllm-project/recipes":
            preserved_entries.append(
                {
                    "alias": entry.alias,
                    "status": entry.status,
                    "canonical_family": entry.canonical_family,
                    "note": entry.note,
                    "suggestions": list(entry.suggestions),
                    "source": entry.source,
                }
            )

    merged = preserved_entries + imported_entries
    merged.sort(key=lambda item: item["alias"])
    output_path.write_text(f"{json.dumps(merged, indent=2)}\n", encoding="utf-8")
    return {
        "output_path": str(output_path),
        "imported_count": len(imported_entries),
        "preserved_count": len(preserved_entries),
    }
