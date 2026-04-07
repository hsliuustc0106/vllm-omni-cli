"""ReAct Agent: Reasoning + Acting loop base class."""

from __future__ import annotations

import logging
from typing import Any

from .agent import BaseAgent, AgentResult
from .context import Context
from .llm import LLMBackend
from .tool import BaseTool, ToolRegistry

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """Agent with ReAct (Reasoning + Acting) loop.

    Implements a think → act → observe loop:
    1. Think: LLM reasons about current state and decides next action
    2. Act: Execute a tool (or delegate to another agent)
    3. Observe: Process tool result
    4. Repeat until finish or max iterations
    """

    def __init__(
        self,
        name: str | None = None,
        role: str | None = None,
        model: str | None = None,
        skills: list | None = None,
        tools: list[BaseTool] | None = None,
        llm: LLMBackend | None = None,
        max_iterations: int = 20,
    ) -> None:
        super().__init__(name=name, role=role, model=model, skills=skills, tools=tools, llm=llm)
        self._max_iterations = max_iterations

    def _get_max_iterations(self) -> int:
        return self._max_iterations

    async def run(self, task: str, ctx: Context | None = None) -> AgentResult:
        """Run ReAct loop: reason → act → observe."""
        if ctx is None:
            from .task import Task
            ctx = Context(task=Task(description=task))

        system = self._build_system_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        openai_tools = self._tool_registry.to_openai_tools()
        iterations = 0

        while iterations < self._get_max_iterations():
            iterations += 1
            logger.debug(f"[{self.name}] ReAct iteration {iterations}/{self._get_max_iterations()}")

            # Think: call LLM
            response = await self._llm.complete(messages, tools=openai_tools or None)

            # Build assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}

            if not response.tool_calls:
                messages.append(assistant_msg)
                ctx.add_message(self.name, response.content or "")
                return AgentResult(content=response.content or "", success=True)

            # Act: execute tool calls
            tc_dicts = []
            for tc in response.tool_calls:
                tc_dicts.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function_name, "arguments": tc.arguments},
                })

                # Observe: execute tool and get result
                result = await self._execute_tool(tc)

                # Handle thinking markers (DeepSeek style)
                if result and any(m in result for m in ["🤔"]):
                    result, _ = self._llm.split_think(result)

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                ctx.add_message(self.name, result, tool_calls=[tc_dicts[-1]])
                logger.info(f"[{self.name}] Tool {tc.function_name} → {len(result)} chars")

            assistant_msg["tool_calls"] = tc_dicts
            messages.append(assistant_msg)

        logger.warning(f"[{self.name}] Reached max iterations ({self._get_max_iterations()})")
        return AgentResult(
            content=f"Agent '{self.name}' reached maximum iterations ({self._get_max_iterations()}). "
                    "Consider increasing max_iterations or breaking down the task.",
            success=False,
        )

    async def run_single_turn(self, task: str, ctx: Context | None = None) -> AgentResult:
        """Single-turn execution (no tool loop). Good for simple queries."""
        if ctx is None:
            from .task import Task
            ctx = Context(task=Task(description=task))

        system = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        response = await self._llm.complete(messages)
        ctx.add_message(self.name, response.content or "")
        return AgentResult(content=response.content or "", success=True)
