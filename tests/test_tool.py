"""Tests for tool registry and base tools."""

import pytest

from vomni.core.tool import BaseTool, ToolRegistry


class DummyTool(BaseTool):
    name = "dummy"
    description = "A dummy tool for testing."
    parameters = {"type": "object", "properties": {"x": {"type": "integer"}}}

    async def execute(self, **kwargs):
        return kwargs.get("x", 0) * 2


@pytest.mark.asyncio
async def test_tool_execute():
    tool = DummyTool()
    result = await tool.execute(x=5)
    assert result == 10


def test_tool_openai_format():
    tool = DummyTool()
    schema = tool.to_openai_tool()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dummy"


def test_tool_anthropic_format():
    tool = DummyTool()
    schema = tool.to_anthropic_tool()
    assert schema["name"] == "dummy"
    assert "input_schema" in schema


def test_registry_register_and_get():
    reg = ToolRegistry()
    reg.register(DummyTool())
    assert reg.get("dummy") is not None
    assert reg.get("nonexistent") is None


def test_registry_to_openai_tools():
    reg = ToolRegistry()
    reg.register(DummyTool())
    tools = reg.to_openai_tools()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "dummy"
