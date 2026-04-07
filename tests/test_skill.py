"""Tests for skill adapter."""

import pytest
from pathlib import Path

from vllm_omni_cli.core.skill import BaseSkill, SkillAdapter
from vllm_omni_cli.core.context import Context, Task


class FakeSkill(BaseSkill):
    name = "fake"
    description = "Fake skill"
    tools = []
    knowledge = "You know things."

    async def run(self, ctx, **kwargs):
        return "ok"


@pytest.mark.asyncio
async def test_fake_skill():
    skill = FakeSkill()
    ctx = Context(task=Task(description="test"))
    result = await skill.run(ctx)
    assert result == "ok"


def test_parse_skill_from_directory(tmp_path: Path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: A test skill\ntools: [shell]\n---\n"
        "# My Skill\n\nSome knowledge here.",
        encoding="utf-8",
    )
    skill = SkillAdapter.load_from_directory(skill_dir)
    assert skill.name == "my-skill"
    assert skill.description == "A test skill"
    assert skill.tools == ["shell"]
    assert "Some knowledge here" in skill.knowledge


def test_load_from_repo(tmp_path: Path):
    for name in ["alpha", "beta"]:
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(f"# {name}\nKnowledge for {name}.", encoding="utf-8")

    skills = SkillAdapter.load_from_repo(tmp_path)
    assert len(skills) == 2
    assert {s.name for s in skills} == {"alpha", "beta"}


# --- New tests ---


def test_skill_without_front_matter_name_from_heading(tmp_path: Path):
    skill_dir = tmp_path / "heading-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "# My Heading Skill\n\nBody content here.", encoding="utf-8"
    )
    skill = SkillAdapter.load_from_directory(skill_dir)
    assert skill.name == "My Heading Skill"
    assert "Body content here" in skill.knowledge


def test_skill_without_front_matter_no_heading(tmp_path: Path):
    skill_dir = tmp_path / "fallback-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("Just plain text without any headings.", encoding="utf-8")
    skill = SkillAdapter.load_from_directory(skill_dir)
    assert skill.name == "fallback-skill"  # directory name as fallback
    assert "plain text" in skill.knowledge


def test_skill_no_tools_field(tmp_path: Path):
    skill_dir = tmp_path / "no-tools"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: tool-less\ndescription: No tools here\n---\nSome knowledge.\n",
        encoding="utf-8",
    )
    skill = SkillAdapter.load_from_directory(skill_dir)
    assert skill.name == "tool-less"
    assert skill.tools == []


def test_load_from_directory_missing_skill_md(tmp_path: Path):
    skill_dir = tmp_path / "empty"
    skill_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No SKILL.md"):
        SkillAdapter.load_from_directory(skill_dir)


def test_load_from_repo_empty_directory(tmp_path: Path):
    empty = tmp_path / "empty-repo"
    empty.mkdir()
    skills = SkillAdapter.load_from_repo(empty)
    assert skills == []


def test_load_from_repo_nonexistent_path(tmp_path: Path):
    skills = SkillAdapter.load_from_repo(tmp_path / "no-such-dir")
    assert skills == []


def test_skill_with_references_subdir(tmp_path: Path):
    """References subdirectory is ignored by load_from_directory (no error)."""
    skill_dir = tmp_path / "ref-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: ref-skill\n---\nKnowledge.\n", encoding="utf-8")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "doc.txt").write_text("ref content", encoding="utf-8")
    skill = SkillAdapter.load_from_directory(skill_dir)
    assert skill.name == "ref-skill"
    assert "Knowledge" in skill.knowledge
