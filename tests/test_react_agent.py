"""Tests for ReActAgent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_omni_cli.core.agent import AgentResult
from vllm_omni_cli.core.context import Context, Task
from vllm_omni_cli.core.llm import LLMResponse, ToolCall, LLMBackend
from vllm_omni_cli.core.react_agent import ReActAgent
from vllm_omni_cli.core.tool import BaseTool


def _make_mock_llm(responses):
    llm = MagicMock(spec=LLMBackend)
    llm.complete = AsyncMock()
    llm.complete.side_effect = responses
    llm._diagnose_output = MagicMock()
    llm.split_think = staticmethod(LLMBackend.split_think)
    return llm


def _tool_call(name, arguments, id="call_1"):
    return ToolCall(id=id, function_name=name, arguments=json.dumps(arguments))


class TestReActAgentInit:
    def test_react_agent_init(self):
        agent = ReActAgent(name="test", role="Test role.")
        assert agent.name == "test"
        assert agent.role == "Test role."
        assert agent._max_iterations == 40

    def test_react_agent_custom_max_iterations(self):
        agent = ReActAgent(name="test", max_iterations=5)
        assert agent._get_max_iterations() == 5


class TestReActAgentNoToolCall:
    @pytest.mark.asyncio
    async def test_react_agent_no_tool_call(self):
        mock_llm = _make_mock_llm([
            LLMResponse(content="Hello! Here is my answer."),
        ])
        agent = ReActAgent(name="test", role="Test.", llm=mock_llm)
        ctx = Context(task=Task(description="Say hello"))
        result = await agent.run("Say hello", ctx)
        assert result.success is True
        assert result.content == "Hello! Here is my answer."


class TestReActAgentSingleToolCall:
    @pytest.mark.asyncio
    async def test_react_agent_single_tool_call(self):
        mock_llm = _make_mock_llm([
            LLMResponse(
                content="",
                tool_calls=[_tool_call("search", {"query": "test"})],
            ),
            LLMResponse(content="Found results."),
        ])
        agent = ReActAgent(name="test", role="Test.", llm=mock_llm)
        ctx = Context(task=Task(description="Search"))
        result = await agent.run("Search for test", ctx)
        assert result.success is True
        assert result.content == "Found results."
        assert mock_llm.complete.call_count == 2


class TestReActAgentMaxIterations:
    @pytest.mark.asyncio
    async def test_react_agent_max_iterations(self):
        tool_responses = [
            LLMResponse(
                content="",
                tool_calls=[_tool_call("search", {"query": f"q{i}"})],
            )
            for i in range(3)
        ]
        mock_llm = _make_mock_llm(tool_responses)
        agent = ReActAgent(name="test", role="Test.", llm=mock_llm, max_iterations=2)
        ctx = Context(task=Task(description="Loop"))
        result = await agent.run("Loop", ctx)
        assert result.success is False
        assert "maximum iterations" in result.content
        assert mock_llm.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_react_agent_stops_on_repeated_tool_call(self):
        repeated = LLMResponse(
            content="",
            tool_calls=[_tool_call("search", {"query": "same"})],
        )
        mock_llm = _make_mock_llm([repeated, repeated, repeated])
        agent = ReActAgent(name="test", role="Test.", llm=mock_llm, max_iterations=10)
        ctx = Context(task=Task(description="Loop"))

        result = await agent.run("Loop", ctx)

        assert result.success is False
        assert "repeating the same tool call" in result.content
        assert mock_llm.complete.call_count == 3


class TestReActAgentSingleTurn:
    @pytest.mark.asyncio
    async def test_react_agent_single_turn(self):
        mock_llm = _make_mock_llm([
            LLMResponse(content="Quick answer."),
        ])
        agent = ReActAgent(name="test", role="Test.", llm=mock_llm)
        result = await agent.run_single_turn("Quick question")
        assert result.success is True
        assert result.content == "Quick answer."
        assert mock_llm.complete.call_count == 1

        sent_messages = mock_llm.complete.call_args.kwargs["messages"]
        assert "Quick Response Mode" in sent_messages[0]["content"]
        assert "Respond with a direct draft answer now." in sent_messages[1]["content"]

class TestReActAgentThinkMarker:
    @pytest.mark.asyncio
    async def test_react_agent_think_marker_handling(self):
        mock_llm = _make_mock_llm([
            LLMResponse(
                content="",
                tool_calls=[_tool_call("think", {"thought": "🤔 Let me think"})],
            ),
            LLMResponse(content="Done."),
        ])
        agent = ReActAgent(name="test", role="Test.", llm=mock_llm)
        ctx = Context(task=Task(description="Think"))
        result = await agent.run("Think about it", ctx)
        assert result.success is True
