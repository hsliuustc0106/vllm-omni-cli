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
    assert agent.model == "glm-5.1"


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


# --- Registry tests ---


def test_register_agent_decorator():
    from vllm_omni_cli.core.registry import AgentRegistry, register_agent

    AgentRegistry.clear()

    @register_agent
    class DummyAgent(BaseAgent):
        def __init__(self, **kw):
            super().__init__(name="dummy", **kw)

    assert AgentRegistry.get("DummyAgent") is DummyAgent
    AgentRegistry.clear()


def test_register_agent_with_name_and_scopes():
    from vllm_omni_cli.core.registry import AgentRegistry, register_agent

    AgentRegistry.clear()

    @register_agent("my-agent", scopes=["hpc", "inference"])
    class CustomAgent(BaseAgent):
        def __init__(self, **kw):
            super().__init__(name="my-agent", **kw)

    assert AgentRegistry.get("my-agent") is CustomAgent
    assert AgentRegistry.list() == ["my-agent"]
    assert "my-agent" in AgentRegistry.list(scope="hpc")
    assert "my-agent" in AgentRegistry.list(scope="inference")
    assert "my-agent" not in AgentRegistry.list(scope="devops")
    AgentRegistry.clear()


def test_registry_list_with_scope():
    from vllm_omni_cli.core.registry import AgentRegistry, register_agent

    AgentRegistry.clear()

    @register_agent("a1", scopes=["hpc"])
    class A1(BaseAgent):
        def __init__(self, **kw): super().__init__(**kw)

    @register_agent("a2", scopes=["dev"])
    class A2(BaseAgent):
        def __init__(self, **kw): super().__init__(**kw)

    @register_agent("a3")  # no scopes → empty set → matched by any scope
    class A3(BaseAgent):
        def __init__(self, **kw): super().__init__(**kw)

    all_list = AgentRegistry.list()
    assert set(all_list) == {"a1", "a2", "a3"}
    assert set(AgentRegistry.list(scope="hpc")) == {"a1", "a3"}
    assert set(AgentRegistry.list(scope="dev")) == {"a2", "a3"}
    AgentRegistry.clear()


def test_registry_create():
    from vllm_omni_cli.core.registry import AgentRegistry, register_agent

    AgentRegistry.clear()

    @register_agent("maker")
    class Maker(BaseAgent):
        def __init__(self, **kw):
            super().__init__(name="maker", **kw)

    agent = AgentRegistry.create("maker")
    assert agent.name == "maker"
    AgentRegistry.clear()


def test_registry_get_unknown():
    from vllm_omni_cli.core.registry import AgentRegistry

    AgentRegistry.clear()
    assert AgentRegistry.get("nonexistent") is None
    with pytest.raises(ValueError, match="not registered"):
        AgentRegistry.create("nonexistent")


def test_registry_discover():
    from unittest.mock import patch, MagicMock
    from vllm_omni_cli.core.registry import AgentRegistry
    from vllm_omni_cli.agents import ArchitectAgent

    AgentRegistry.clear()

    mock_ep = MagicMock()
    mock_ep.name = "discovered-agent"
    mock_ep.load.return_value = ArchitectAgent

    with patch("vllm_omni_cli.core.registry.importlib.metadata.entry_points") as mock_eps:
        mock_eps.return_value = [mock_ep]
        count = AgentRegistry.discover()
    assert count == 1
    assert AgentRegistry.get("discovered-agent") is ArchitectAgent
    AgentRegistry.clear()
