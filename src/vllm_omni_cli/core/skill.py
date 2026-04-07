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
    category: str = ""  # e.g. "workflow", "guide", "agent"
    version: str = "1.0.0"

    @abstractmethod
    async def run(self, ctx: Context, **kwargs: Any) -> Any: ...


class SkillMetadata:
    """Metadata for a loaded skill."""

    def __init__(self, name: str, description: str, knowledge: str,
                 tools: list[str], category: str = "", version: str = "1.0.0",
                 source_path: str = ""):
        self.name = name
        self.description = description
        self.knowledge = knowledge
        self.tools = tools
        self.category = category
        self.version = version
        self.source_path = source_path

    def validate(self) -> tuple[bool, str]:
        if not self.name:
            return False, "name is required"
        if not self.knowledge:
            return False, "knowledge is required"
        return True, ""

    def __repr__(self):
        return f"<Skill {self.name} v{self.version}>"


class SkillRegistry:
    """Skill registry with version management and category indexing."""

    def __init__(self):
        self._skills: dict[str, SkillMetadata] = {}  # name -> latest version
        self._all_versions: dict[str, dict[str, SkillMetadata]] = {}  # name -> {version: metadata}
        self._by_category: dict[str, list[SkillMetadata]] = {}

    def register(self, skill: SkillMetadata) -> bool:
        valid, err = skill.validate()
        if not valid:
            return False

        self._all_versions.setdefault(skill.name, {})[skill.version] = skill

        current = self._skills.get(skill.name)
        if not current or self._is_newer(skill.version, current.version):
            self._skills[skill.name] = skill
            if current and current.category:
                cats = self._by_category.get(current.category, [])
                if current in cats:
                    cats.remove(current)
            if skill.category:
                self._by_category.setdefault(skill.category, []).append(skill)

        return True

    def get(self, name: str, version: str | None = None) -> SkillMetadata | None:
        if version:
            return self._all_versions.get(name, {}).get(version)
        return self._skills.get(name)

    def get_all(self) -> list[SkillMetadata]:
        return list(self._skills.values())

    def get_by_category(self, category: str) -> list[SkillMetadata]:
        return list(self._by_category.get(category, []))

    def get_versions(self, name: str) -> list[str]:
        versions = list(self._all_versions.get(name, {}).keys())
        versions.sort(key=lambda v: [int(x) for x in v.split('.')])
        return versions

    def exists(self, name: str) -> bool:
        return name in self._skills

    def unregister(self, name: str) -> bool:
        if name not in self._skills:
            return False
        skill = self._skills.pop(name)
        self._all_versions.pop(name, None)
        if skill.category and skill.category in self._by_category:
            cats = self._by_category[skill.category]
            if skill in cats:
                cats.remove(skill)
        return True

    def load_from_directory(self, skill_dir: Path) -> int:
        """Load all skills from a directory containing SKILL.md subdirs."""
        count = 0
        if not skill_dir.exists():
            return count
        for child in sorted(skill_dir.iterdir()):
            if child.is_dir() and (child / "SKILL.md").exists():
                skill = SkillAdapter.load_from_directory(child)
                meta = SkillMetadata(
                    name=skill.name, description=skill.description,
                    knowledge=skill.knowledge, tools=skill.tools,
                    category=skill.category, version=skill.version,
                    source_path=str(child),
                )
                if self.register(meta):
                    count += 1
        return count

    def load_single(self, skill_path: Path) -> bool:
        """Load a single skill from its directory (containing SKILL.md)."""
        skill = SkillAdapter.load_from_directory(skill_path)
        meta = SkillMetadata(
            name=skill.name, description=skill.description,
            knowledge=skill.knowledge, tools=skill.tools,
            category=skill.category, version=skill.version,
            source_path=str(skill_path),
        )
        return self.register(meta)

    def filter(self, category: str | None = None, name_pattern: str | None = None) -> list[SkillMetadata]:
        import fnmatch
        skills = self.get_by_category(category) if category else self.get_all()
        if name_pattern:
            skills = [s for s in skills if fnmatch.fnmatch(s.name, name_pattern)]
        return skills

    def get_statistics(self) -> dict:
        return {
            "total": len(self._skills),
            "total_versions": sum(len(v) for v in self._all_versions.values()),
            "categories": {cat: len(skills) for cat, skills in self._by_category.items()},
        }

    @staticmethod
    def _is_newer(v1: str, v2: str) -> bool:
        try:
            parts1 = [int(x) for x in v1.split('.')]
            parts2 = [int(x) for x in v2.split('.')]
            return parts1 > parts2
        except ValueError:
            return v1 > v2

    def clear(self):
        self._skills.clear()
        self._all_versions.clear()
        self._by_category.clear()

    def __len__(self):
        return len(self._skills)

    def __contains__(self, name: str):
        return name in self._skills


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
        category = ""
        version = "1.0.0"

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
                elif line.startswith("category:"):
                    category = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("version:"):
                    version = line.split(":", 1)[1].strip().strip('"\'')
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
        skill.category = category
        skill.version = version
        return skill
