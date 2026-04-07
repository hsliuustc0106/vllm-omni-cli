"""Tests for skill adapter."""

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
