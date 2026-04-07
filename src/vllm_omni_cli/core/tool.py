"""Tool abstraction and registry."""

from __future__ import annotations

import importlib.metadata
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseTool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    category: str = "basic"  # basic, domain, execution, interaction
    scopes: list[str] = ["all"]  # which agents can use this tool

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any: ...

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class ToolRegistry:
    """Registry for discovering and managing tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def discover(self) -> None:
        """Discover tools via importlib.metadata entry_points 'vllm_omni_cli.tools'."""
        try:
            eps = importlib.metadata.entry_points(group="vllm_omni_cli.tools")
        except AttributeError:
            return
        for ep in eps:
            try:
                cls = ep.load()
                if isinstance(cls, type) and issubclass(cls, BaseTool) and cls is not BaseTool:
                    instance = cls()
                    self.register(instance)
            except Exception:
                continue

    def get_tools(self, scope: str | None = None, categories: list[str] | None = None) -> list[BaseTool]:
        tools = []
        for tool in self._tools.values():
            if scope and "all" not in tool.scopes and scope not in tool.scopes:
                continue
            if categories and tool.category not in categories:
                continue
            tools.append(tool)
        return tools

    def to_openai_tools(self, scope: str | None = None, categories: list[str] | None = None) -> list[dict[str, Any]]:
        return [t.to_openai_tool() for t in self.get_tools(scope=scope, categories=categories)]

    def to_markdown(self, scope: str | None = None) -> str:
        """Format tools as markdown for prompt injection."""
        tools = self.get_tools(scope=scope)
        lines = []
        for tool in tools:
            params = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])
            param_strs = []
            for pname, pinfo in params.items():
                req = " (required)" if pname in required else " (optional)"
                desc = pinfo.get("description", pinfo.get("type", ""))
                param_strs.append(f"  - {pname}: {desc}{req}")
            lines.append(f"### {tool.name}\n{tool.description}\nParameters:\n" + "\n".join(param_strs))
        return "\n\n".join(lines)

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_tool() for t in self._tools.values()]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
