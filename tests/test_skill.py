"""Tests for skill adapter."""

import pytest
from pathlib import Path

from vllm_omni_cli.core.skill import BaseSkill, SkillAdapter, SkillMetadata, SkillRegistry
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


# --- SkillRegistry tests ---


def test_skill_registry_register_and_get():
    reg = SkillRegistry()
    meta = SkillMetadata(name="a", description="desc", knowledge="k", tools=[])
    assert reg.register(meta)
    assert reg.get("a") is meta
    assert reg.get("nonexistent") is None
    assert "a" in reg
    assert len(reg) == 1


def test_skill_registry_version_management():
    reg = SkillRegistry()
    v1 = SkillMetadata(name="a", description="desc", knowledge="k", tools=[], version="1.0.0")
    v2 = SkillMetadata(name="a", description="desc", knowledge="k", tools=[], version="2.1.0")
    reg.register(v1)
    reg.register(v2)
    assert reg.get("a") is v2  # latest
    assert reg.get("a", version="1.0.0") is v1
    assert reg.get_versions("a") == ["1.0.0", "2.1.0"]


def test_skill_registry_category_indexing():
    reg = SkillRegistry()
    reg.register(SkillMetadata(name="a", description="", knowledge="k", tools=[], category="guide"))
    reg.register(SkillMetadata(name="b", description="", knowledge="k", tools=[], category="guide"))
    reg.register(SkillMetadata(name="c", description="", knowledge="k", tools=[], category="agent"))
    assert len(reg.get_by_category("guide")) == 2
    assert len(reg.get_by_category("agent")) == 1


def test_skill_registry_filter():
    reg = SkillRegistry()
    reg.register(SkillMetadata(name="foo-bar", description="", knowledge="k", tools=[], category="guide"))
    reg.register(SkillMetadata(name="foo-baz", description="", knowledge="k", tools=[], category="guide"))
    reg.register(SkillMetadata(name="qux", description="", knowledge="k", tools=[], category="agent"))
    assert len(reg.filter(category="guide")) == 2
    assert len(reg.filter(name_pattern="foo-*")) == 2
    assert len(reg.filter(category="guide", name_pattern="foo-baz")) == 1


def test_skill_registry_statistics():
    reg = SkillRegistry()
    reg.register(SkillMetadata(name="a", description="", knowledge="k", tools=[], category="g", version="1.0.0"))
    reg.register(SkillMetadata(name="a", description="", knowledge="k", tools=[], category="g", version="2.0.0"))
    reg.register(SkillMetadata(name="b", description="", knowledge="k", tools=[], category="a"))
    stats = reg.get_statistics()
    assert stats["total"] == 2
    assert stats["total_versions"] == 3
    assert stats["categories"] == {"g": 1, "a": 1}


def test_skill_registry_load_from_directory(tmp_path: Path):
    for name in ["s1", "s2"]:
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(f"---\nname: {name}\ncategory: test\n---\nKnowledge for {name}.", encoding="utf-8")
    reg = SkillRegistry()
    count = reg.load_from_directory(tmp_path)
    assert count == 2
    assert len(reg) == 2


def test_skill_registry_unregister():
    reg = SkillRegistry()
    reg.register(SkillMetadata(name="a", description="", knowledge="k", tools=[], category="guide"))
    assert reg.unregister("a")
    assert not reg.exists("a")
    assert len(reg.get_by_category("guide")) == 0
    assert not reg.unregister("a")  # already gone


def test_skill_metadata_validation():
    valid = SkillMetadata(name="a", description="", knowledge="k", tools=[])
    assert valid.validate() == (True, "")
    no_name = SkillMetadata(name="", description="", knowledge="k", tools=[])
    assert no_name.validate() == (False, "name is required")
    no_knowledge = SkillMetadata(name="a", description="", knowledge="", tools=[])
    assert no_knowledge.validate() == (False, "knowledge is required")
    assert not SkillRegistry().register(no_name)


def test_skill_adapter_extracts_version_and_category(tmp_path: Path):
    d = tmp_path / "v-skill"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: vs\ncategory: guide\nversion: 2.0.0\n---\nKnowledge.\n", encoding="utf-8"
    )
    skill = SkillAdapter.load_from_directory(d)
    assert skill.category == "guide"
    assert skill.version == "2.0.0"


def test_skill_adapter_fallback_version(tmp_path: Path):
    d = tmp_path / "no-ver"
    d.mkdir()
    (d / "SKILL.md").write_text("# No Version\nBody.\n", encoding="utf-8")
    skill = SkillAdapter.load_from_directory(d)
    assert skill.version == "1.0.0"
    assert skill.category == ""
