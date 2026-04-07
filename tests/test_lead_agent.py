"""Tests for LeadAgent orchestrator."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_omni_cli.core.agent import AgentResult, BaseAgent
from vllm_omni_cli.core.context import Context, Task
from vllm_omni_cli.core.lead_agent import LeadAgent, LeadAgentResult
from vllm_omni_cli.core.llm import LLMResponse, ToolCall


def _make_mock_llm(responses):
    """Create a mock LLMBackend that returns responses in sequence."""
    llm = MagicMock()
    llm.complete = AsyncMock()
    llm.complete.side_effect = responses
    return llm


def _make_agent(name="test_agent", role="Test agent role.") -> BaseAgent:
    agent = BaseAgent(name=name, role=role)
    agent.run = AsyncMock(return_value=AgentResult(content=f"Result from {name}", success=True))
    return agent


def _tool_call(name, arguments, id="call_1"):
    return ToolCall(id=id, function_name=name, arguments=json.dumps(arguments))


class TestLeadAgentInit:
    def test_lead_agent_init(self):
        agents = [_make_agent("a1"), _make_agent("a2")]
        lead = LeadAgent(agents=agents)
        assert "a1" in lead.agents
        assert "a2" in lead.agents
        assert lead.max_rounds == 20
        assert lead.human_in_the_loop is False

    def test_lead_agent_custom_settings(self):
        lead = LeadAgent(agents=[], max_rounds=5, human_in_the_loop=True)
        assert lead.max_rounds == 5
        assert lead.human_in_the_loop is True


class TestLeadAgentSystemPrompt:
    def test_contains_agent_descriptions(self):
        agents = [_make_agent("architect", "HPC architect for distributed inference.")]
        lead = LeadAgent(agents=agents)
        prompt = lead._build_system_prompt()
        assert "architect" in prompt
        assert "HPC architect" in prompt
        assert "Lead Agent" in prompt
        assert "delegate" in prompt.lower()
        assert "finish" in prompt


class TestLeadAgentTools:
    def test_tools_defined(self):
        agents = [_make_agent("coder")]
        lead = LeadAgent(agents=agents)
        tools = lead._build_tools()
        tool_names = [t["function"]["name"] for t in tools]
        assert "delegate_agent" in tool_names
        assert "ask_user" in tool_names
        assert "finish" in tool_names

    def test_delegate_agent_enum(self):
        agents = [_make_agent("coder"), _make_agent("optimizer")]
        lead = LeadAgent(agents=agents)
        tools = lead._build_tools()
        delegate_tool = next(t for t in tools if t["function"]["name"] == "delegate_agent")
        enum = delegate_tool["function"]["parameters"]["properties"]["agent_name"]["enum"]
        assert "coder" in enum
        assert "optimizer" in enum


class TestLeadAgentDelegate:
    @pytest.mark.asyncio
    async def test_delegate_success(self):
        mock_llm = _make_mock_llm([
            LLMResponse(
                content="",
                tool_calls=[_tool_call("delegate_agent", {"agent_name": "coder", "task_description": "Write code"})],
            ),
            LLMResponse(
                content="",
                tool_calls=[_tool_call("finish", {"summary": "All done"})],
            ),
        ])
        agent = _make_agent("coder")
        lead = LeadAgent(agents=[agent], llm=mock_llm)
        result = await lead.run("Write some code")

        assert result.success is True
        assert result.content == "All done"
        assert result.rounds == 2
        assert len(result.agent_calls) == 1
        assert result.agent_calls[0]["agent_name"] == "coder"
        agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_agent(self):
        mock_llm = _make_mock_llm([
            LLMResponse(
                content="",
                tool_calls=[_tool_call("delegate_agent", {"agent_name": "nonexistent", "task_description": "Do stuff"})],
            ),
            LLMResponse(
                content="",
                tool_calls=[_tool_call("finish", {"summary": "Failed"})],
            ),
        ])
        lead = LeadAgent(agents=[], llm=mock_llm)
        result = await lead.run("Do stuff")
        assert result.success is True  # finish was called
        assert result.success is True  # finish was called after error


class TestLeadAgentFinish:
    @pytest.mark.asyncio
    async def test_finish_ends_loop(self):
        mock_llm = _make_mock_llm([
            LLMResponse(
                content="",
                tool_calls=[_tool_call("finish", {"summary": "Task complete!"})],
            ),
        ])
        lead = LeadAgent(agents=[], llm=mock_llm)
        result = await lead.run("Some task")
        assert result.content == "Task complete!"
        assert result.rounds == 1
        assert mock_llm.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_no_tool_call_treated_as_finish(self):
        mock_llm = _make_mock_llm([
            LLMResponse(content="Done! No tool needed."),
        ])
        lead = LeadAgent(agents=[], llm=mock_llm)
        result = await lead.run("Simple task")
        assert result.content == "Done! No tool needed."
        assert result.success is True


class TestLeadAgentMaxRounds:
    @pytest.mark.asyncio
    async def test_max_rounds(self):
        # Always delegate, never finish
        mock_llm = _make_mock_llm([
            LLMResponse(
                content="",
                tool_calls=[_tool_call("delegate_agent", {"agent_name": "coder", "task_description": f"Task {i}"})],
            )
            for i in range(5)
        ])
        agent = _make_agent("coder")
        lead = LeadAgent(agents=[agent], llm=mock_llm, max_rounds=3)
        result = await lead.run("Loop task")
        assert result.success is False
        assert result.rounds == 3
        assert "Max rounds" in result.content
