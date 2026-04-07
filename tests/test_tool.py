"""Tests for tool registry and base tools."""

import pytest

from vllm_omni_cli.core.tool import BaseTool, ToolRegistry
from vllm_omni_cli.tools.github import GitHubTool
from vllm_omni_cli.tools.vllm import VllmTool
from vllm_omni_cli.tools.shell import ShellTool


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


# --- New tests ---


class ComplexTool(BaseTool):
    name = "complex"
    description = "A tool with complex JSON Schema."
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["fast", "accurate", "balanced"],
            },
            "config": {
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer", "default": 32},
                    "dtype": {"type": "string", "enum": ["float16", "bfloat16", "float32"]},
                },
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["mode"],
    }

    async def execute(self, **kwargs):
        return kwargs


def test_complex_tool_schema():
    """Tool with nested objects, arrays, enums preserves full schema."""
    tool = ComplexTool()
    assert tool.parameters["properties"]["mode"]["enum"] == ["fast", "accurate", "balanced"]
    assert "items" in tool.parameters["properties"]["tags"]
    assert tool.parameters["properties"]["config"]["type"] == "object"


def test_duplicate_registration_overwrites():
    """Registering a tool with the same name overwrites the previous one."""

    class ToolA(BaseTool):
        name = "dup"
        async def execute(self, **kw):
            return "a"

    class ToolB(BaseTool):
        name = "dup"
        async def execute(self, **kw):
            return "b"

    reg = ToolRegistry()
    reg.register(ToolA())
    reg.register(ToolB())
    tool = reg.get("dup")
    assert tool.execute.__self__ is not None  # just check it's the second one
    assert isinstance(tool, ToolB)


def test_get_unregistered_returns_none():
    reg = ToolRegistry()
    assert reg.get("nope") is None


@pytest.mark.parametrize("cls", [GitHubTool, VllmTool, ShellTool])
def test_builtin_tools_have_required_fields(cls):
    tool = cls()
    assert isinstance(tool.name, str) and len(tool.name) > 0
    assert isinstance(tool.description, str) and len(tool.description) > 0
    assert tool.parameters["type"] == "object"
    assert "properties" in tool.parameters


def test_to_openai_tools_format():
    reg = ToolRegistry()
    reg.register(GitHubTool())
    tools = reg.to_openai_tools()
    assert isinstance(tools, list) and len(tools) == 1
    t = tools[0]
    assert t["type"] == "function"
    assert set(t["function"].keys()) >= {"name", "description", "parameters"}
    assert t["function"]["parameters"]["type"] == "object"


def test_to_anthropic_tools_format():
    reg = ToolRegistry()
    reg.register(ShellTool())
    tools = reg.to_anthropic_tools()
    assert isinstance(tools, list) and len(tools) == 1
    t = tools[0]
    assert set(t.keys()) >= {"name", "description", "input_schema"}
    assert t["input_schema"]["type"] == "object"
