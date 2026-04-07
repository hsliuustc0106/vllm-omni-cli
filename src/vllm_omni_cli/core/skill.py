"""Skill abstraction and SKILL.md adapter."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .context import Context


class BaseSkill(ABC):
    """Base class for skills."""

    name: str = ""
    description: str = ""
    tools: list[str] = []
    knowledge: str = ""

    @abstractmethod
    async def run(self, ctx: Context, **kwargs: Any) -> Any: ...


class SkillAdapter:
    """Load skills from SKILL.md files (vllm-omni-skills format)."""

    @staticmethod
    def load_from_directory(path: Path) -> BaseSkill:
        """Parse a single skill directory containing SKILL.md."""
        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"No SKILL.md found in {path}")

        text = skill_md.read_text(encoding="utf-8")
        return SkillAdapter._parse_skill(text, path.name)

    @staticmethod
    def load_from_repo(repo_path: Path) -> list[BaseSkill]:
        """Scan a repo for all skill directories (each containing SKILL.md)."""
        skills: list[BaseSkill] = []
        if not repo_path.exists():
            return skills
        for child in sorted(repo_path.iterdir()):
            if child.is_dir() and (child / "SKILL.md").exists():
                skills.append(SkillAdapter.load_from_directory(child))
        return skills

    @staticmethod
    def _parse_skill(text: str, fallback_name: str) -> BaseSkill:
        """Parse SKILL.md content into a BaseSkill instance."""

        class _ParsedSkill(BaseSkill):
            async def run(self, ctx: Context, **kwargs: Any) -> Any:
                return self.knowledge

        # Extract YAML front matter if present
        knowledge_parts: list[str] = []
        name = fallback_name
        description = ""
        tools: list[str] = []

        # Try to extract from fenced YAML block
        yaml_match = re.search(r"^---\s*\n(.*?)\n---", text, re.DOTALL | re.MULTILINE)
        if yaml_match:
            yaml_block = yaml_match.group(1)
            for line in yaml_block.splitlines():
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("tools:"):
                    tools_str = line.split(":", 1)[1].strip()
                    if tools_str.startswith("["):
                        import ast
                        try:
                            tools = ast.literal_eval(tools_str)
                        except Exception:
                            tools = [t.strip().strip('"\'') for t in tools_str.strip("[]").split(",")]
                    else:
                        tools = [t.strip() for t in tools_str.split(",") if t.strip()]
            # Knowledge is everything after the front matter
            knowledge_parts.append(text[yaml_match.end():].strip())
        else:
            # No front matter — use the whole text as knowledge
            # Try to get name from first heading
            heading = re.match(r"^#\s+(.+)", text, re.MULTILINE)
            if heading:
                name = heading.group(1).strip()
                knowledge_parts.append(text[heading.end():].strip())
            else:
                knowledge_parts.append(text.strip())

        skill = _ParsedSkill()
        skill.name = name
        skill.description = description or (knowledge_parts[0][:120] + "..." if knowledge_parts else "")
        skill.tools = tools
        skill.knowledge = "\n\n".join(knowledge_parts)
        return skill
