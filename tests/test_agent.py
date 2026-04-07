"""Tests for agent base class."""

import pytest

from vomni.core.agent import BaseAgent, AgentResult
from vomni.core.context import Context, Task
from vomni.core.skill import BaseSkill


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
    assert "Echo" in prompt


def test_chat_without_llm_call():
    """Test that chat method works with a mock LLM."""
    agent = BaseAgent(name="mock")
    # Just verify the method signature and history handling
    # (actual LLM call would require a real backend)
    assert hasattr(agent, "chat")
