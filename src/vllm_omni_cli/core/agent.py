"""Agent abstraction with tool-calling loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from .context import Context
from .llm import LLMBackend, ToolCall
from .skill import BaseSkill
from .tool import BaseTool, ToolRegistry


class AgentResult(BaseModel):
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    success: bool = True


class BaseAgent:
    """Base agent with LLM, skills, and tool-calling loop."""

    name: str = "agent"
    role: str = "You are a helpful assistant."
    model: str = "glm-5.1"
    skills: list[BaseSkill] = []
    tools: list[BaseTool] = []

    def __init__(
        self,
        name: str | None = None,
        role: str | None = None,
        model: str | None = None,
        skills: list[BaseSkill] | None = None,
        tools: list[BaseTool] | None = None,
        llm: LLMBackend | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        if role is not None:
            self.role = role
        if model is not None:
            self.model = model
        if skills is not None:
            self.skills = list(skills)
        if tools is not None:
            self.tools = list(tools)
        self._llm = llm or LLMBackend(model=self.model)
        self._progress_callback = progress_callback
        self._tool_registry = ToolRegistry()
        for t in self.tools:
            self._tool_registry.register(t)

    def _emit_progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)

    def _build_system_prompt(self) -> str:
        parts = [self.role]
        for skill in self.skills:
            if skill.knowledge:
                parts.append(f"\n## Skill: {skill.name}\n{skill.knowledge}")
        return "\n".join(parts)

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        tool = self._tool_registry.get(tool_call.function_name)
        if tool is None:
            return f"Error: Tool '{tool_call.function_name}' not found."
        try:
            import json
            args = json.loads(tool_call.arguments) if tool_call.arguments else {}
            args["_agent_name"] = self.name
            result = await tool.execute(**args)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_call.function_name}': {e}"

    async def run(self, task: str, ctx: Context) -> AgentResult:
        """Run the agent on a task with full tool-calling loop."""
        self._emit_progress(f"[{self.name}] starting")
        system = self._build_system_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        openai_tools = self._tool_registry.to_openai_tools(scope=self.name)
        max_iterations = 20

        for _ in range(max_iterations):
            response = await self._llm.complete(messages, tools=openai_tools or None)
            messages.append({"role": "assistant", "content": response.content})

            if not response.tool_calls:
                self._emit_progress(f"[{self.name}] finished")
                return AgentResult(content=response.content, success=True)

            tc_dicts: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                tc_dicts.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function_name, "arguments": tc.arguments},
                })
                self._emit_progress(f"[{self.name}] tool -> {tc.function_name}")
                result = await self._execute_tool(tc)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                ctx.add_message(self.name, result, tool_calls=tc_dicts)

        self._emit_progress(f"[{self.name}] stopped after max iterations")
        return AgentResult(content="Agent reached max tool-call iterations.", success=False)

    async def chat(self, message: str, history: list[dict[str, Any]]) -> str:
        """Single-turn chat with message history."""
        system = self._build_system_prompt()
        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]
        response = await self._llm.complete(messages)
        return response.content
