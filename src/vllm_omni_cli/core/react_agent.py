"""ReAct Agent: Reasoning + Acting loop base class."""

from __future__ import annotations

from collections.abc import Callable
import json
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

    DEFAULT_MAX_ITERATIONS = 40
    REPEATED_TOOL_CALL_LIMIT = 3
    QUICK_RESPONSE_INSTRUCTION = (
        "You are in quick-response mode. Give a useful provisional answer immediately. "
        "Do not say that you need to inspect code, gather more information, or do research first. "
        "State key assumptions briefly, then provide the best concrete recommendation you can. "
        "Prefer concise output with actionable configuration choices. "
        "If a model alias note is present, follow it exactly and restate the resolved family clearly in the answer."
    )

    def __init__(
        self,
        name: str | None = None,
        role: str | None = None,
        model: str | None = None,
        skills: list | None = None,
        tools: list[BaseTool] | None = None,
        llm: LLMBackend | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            role=role,
            model=model,
            skills=skills,
            tools=tools,
            llm=llm,
            progress_callback=progress_callback,
        )
        self._max_iterations = max_iterations

    def _get_max_iterations(self) -> int:
        return self._max_iterations

    def _build_system_prompt(self) -> str:
        base_prompt = super()._build_system_prompt()
        stop_rules = (
            "\n\n## Execution Rules\n"
            "- Use tools only when they are necessary to make progress.\n"
            "- If you already have enough information to answer the task well, stop calling tools and return the answer.\n"
            "- Do not repeat the same tool call with the same arguments unless the previous result explicitly asks you to retry.\n"
            "- After receiving useful tool output, synthesize it into a direct final answer instead of continuing to explore.\n"
            "- If a tool is unavailable or unhelpful, explain that clearly and finish with the best answer you can provide."
        )
        return base_prompt + stop_rules

    @staticmethod
    def _preview_text(content: str, limit: int = 160) -> str:
        normalized = " ".join(content.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    @staticmethod
    def _tool_signature(tool_call: Any) -> str:
        try:
            parsed_args = json.loads(tool_call.arguments) if tool_call.arguments else {}
        except json.JSONDecodeError:
            parsed_args = tool_call.arguments or ""
        canonical_args = json.dumps(parsed_args, sort_keys=True, ensure_ascii=True)
        return f"{tool_call.function_name}:{canonical_args}"

    async def run(self, task: str, ctx: Context | None = None) -> AgentResult:
        """Run ReAct loop: reason → act → observe."""
        if ctx is None:
            from .task import Task
            ctx = Context(task=Task(description=task))

        self._emit_progress(f"[{self.name}] starting")
        system = self._build_system_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        openai_tools = self._tool_registry.to_openai_tools(scope=self.name)
        iterations = 0
        repeated_tool_signature = ""
        repeated_tool_count = 0
        last_tool_result_preview = ""

        while iterations < self._get_max_iterations():
            iterations += 1
            logger.debug(
                "[%s] ReAct iteration %s/%s (messages=%s, tools=%s)",
                self.name,
                iterations,
                self._get_max_iterations(),
                len(messages),
                len(openai_tools),
            )

            # Think: call LLM
            response = await self._llm.complete(messages, tools=openai_tools or None)
            logger.debug(
                "[%s] Iteration %s response: content_len=%s tool_calls=%s preview=%r",
                self.name,
                iterations,
                len(response.content or ""),
                len(response.tool_calls),
                self._preview_text(response.content or ""),
            )

            # Build assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}

            if not response.tool_calls:
                messages.append(assistant_msg)
                ctx.add_message(self.name, response.content or "")
                self._emit_progress(f"[{self.name}] finished")
                return AgentResult(content=response.content or "", success=True)

            # Act: execute tool calls
            tc_dicts = []
            for tc in response.tool_calls:
                signature = self._tool_signature(tc)
                if signature == repeated_tool_signature:
                    repeated_tool_count += 1
                else:
                    repeated_tool_signature = signature
                    repeated_tool_count = 1

                logger.debug(
                    "[%s] Tool call %s/%s: %s args=%s repeat_count=%s",
                    self.name,
                    len(tc_dicts) + 1,
                    len(response.tool_calls),
                    tc.function_name,
                    self._preview_text(tc.arguments or ""),
                    repeated_tool_count,
                )
                self._emit_progress(f"[{self.name}] tool -> {tc.function_name}")

                if repeated_tool_count >= self.REPEATED_TOOL_CALL_LIMIT:
                    logger.warning(
                        "[%s] Stopping after repeated tool call %r (%s consecutive times). "
                        "Last tool result preview=%r",
                        self.name,
                        signature,
                        repeated_tool_count,
                        last_tool_result_preview,
                    )
                    fallback = response.content or (
                        f"Agent '{self.name}' stopped after repeating the same tool call "
                        f"{repeated_tool_count} times. Last tool result: {last_tool_result_preview or '(none)'}"
                    )
                    ctx.add_message(self.name, fallback)
                    self._emit_progress(f"[{self.name}] stopped after repeated tool call")
                    return AgentResult(content=fallback, success=False)

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

                last_tool_result_preview = self._preview_text(result)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                ctx.add_message(self.name, result, tool_calls=[tc_dicts[-1]])
                logger.info(
                    "[%s] Tool %s -> %s chars, result preview=%r",
                    self.name,
                    tc.function_name,
                    len(result),
                    last_tool_result_preview,
                )

            assistant_msg["tool_calls"] = tc_dicts
            messages.append(assistant_msg)

        logger.warning(
            "[%s] Reached max iterations (%s). Last tool result preview=%r",
            self.name,
            self._get_max_iterations(),
            last_tool_result_preview,
        )
        self._emit_progress(f"[{self.name}] stopped after max iterations")
        return AgentResult(
            content=f"Agent '{self.name}' reached maximum iterations ({self._get_max_iterations()}). "
                    f"Last tool result: {last_tool_result_preview or '(none)'}. "
                    "Consider increasing max_iterations or breaking down the task.",
            success=False,
        )

    async def run_single_turn(self, task: str, ctx: Context | None = None) -> AgentResult:
        """Single-turn execution (no tool loop). Good for simple queries."""
        if ctx is None:
            from .task import Task
            ctx = Context(task=Task(description=task))

        self._emit_progress(f"[{self.name}] quick response")
        system = self._build_system_prompt() + "\n\n## Quick Response Mode\n" + self.QUICK_RESPONSE_INSTRUCTION
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"{task}\n\n"
                    "Respond with a direct draft answer now. "
                    "If any facts are uncertain, make reasonable assumptions and label them briefly."
                ),
            },
        ]

        response = await self._llm.complete(messages)
        ctx.add_message(self.name, response.content or "")
        self._emit_progress(f"[{self.name}] quick response ready")
        return AgentResult(content=response.content or "", success=True)
