"""Tests for Context."""

from vllm_omni_cli.core.context import Context, AgentMessage, Artifact
from vllm_omni_cli.core.task import Task


def test_context_creation():
    t = Task(description="do stuff")
    ctx = Context(task=t)
    assert ctx.task.description == "do stuff"
    assert ctx.state == {}
    assert ctx.messages == []
    assert ctx.artifacts == []


def test_context_state_mutations():
    ctx = Context(task=Task(description="t"))
    ctx.state["key"] = "value"
    ctx.state["count"] = 42
    assert ctx.state["key"] == "value"
    assert ctx.state["count"] == 42


def test_context_messages():
    ctx = Context(task=Task(description="t"))
    ctx.add_message("agent-a", "hello")
    ctx.add_message("agent-b", "world", tool_calls=[{"id": "1"}])
    assert len(ctx.messages) == 2
    assert ctx.messages[0].agent_name == "agent-a"
    assert ctx.messages[0].content == "hello"
    assert ctx.messages[1].tool_calls == [{"id": "1"}]


def test_context_artifacts():
    ctx = Context(task=Task(description="t"))
    ctx.add_artifact("result", "analysis", "some output")
    assert len(ctx.artifacts) == 1
    assert ctx.artifacts[0].name == "result"
    assert ctx.artifacts[0].type == "analysis"
    assert ctx.artifacts[0].content == "some output"


def test_agent_message_has_timestamp():
    msg = AgentMessage(agent_name="a", content="b")
    assert msg.timestamp is not None
