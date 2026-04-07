"""Tests for agent base class."""

import pytest

from vllm_omni_cli.core.agent import BaseAgent, AgentResult
from vllm_omni_cli.core.context import Context, Task
from vllm_omni_cli.core.skill import BaseSkill
from vllm_omni_cli.agents import BUILTIN_AGENTS


class EchoSkill(BaseSkill):
    name = "echo"
    description = "Echoes things"
    tools = []
    knowledge = "You are an echo assistant."

    async def run(self, ctx, **kwargs):
        return "echo"


def test_agent_init():
    agent = BaseAgent(name="test", role="Test role.")
    assert agent.name == "test"
    assert agent.role == "Test role."


def test_build_system_prompt():
    agent = BaseAgent(role="Base role", skills=[EchoSkill()])
    prompt = agent._build_system_prompt()
    assert "Base role" in prompt
    assert "echo" in prompt


def test_chat_without_llm_call():
    """Test that chat method works with a mock LLM."""
    agent = BaseAgent(name="mock")
    # Just verify the method signature and history handling
    # (actual LLM call would require a real backend)
    assert hasattr(agent, "chat")


# --- New tests ---


@pytest.mark.parametrize("key, cls", BUILTIN_AGENTS.items())
def test_builtin_agents_exist(key, cls):
    agent = cls()
    assert agent.name == key
    assert len(agent.role) > 0
    assert agent.model == "gpt-4o"


class _SkillA(BaseSkill):
    name = "skill-a"
    knowledge = "Knowledge A"
    tools = []
    description = ""
    async def run(self, ctx, **kw): return "a"

class _SkillB(BaseSkill):
    name = "skill-b"
    knowledge = "Knowledge B"
    tools = []
    description = ""
    async def run(self, ctx, **kw): return "b"


def test_agent_with_multiple_skills():
    s1, s2 = _SkillA(), _SkillB()

    agent = BaseAgent(role="Base", skills=[s1, s2])
    prompt = agent._build_system_prompt()
    assert "Knowledge A" in prompt
    assert "Knowledge B" in prompt


def test_agent_with_tools_populates_registry():
    from vllm_omni_cli.core.tool import BaseTool

    class MyTool(BaseTool):
        name = "my-tool"
        async def execute(self, **kw):
            return "ok"

    agent = BaseAgent(tools=[MyTool()])
    assert agent._tool_registry.get("my-tool") is not None
    assert len(agent._tool_registry.list_tools()) == 1
