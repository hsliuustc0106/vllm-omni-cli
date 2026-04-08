"""LeadAgent: orchestrator that plans and delegates to sub-agents."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .agent import BaseAgent
from .context import Artifact, Context
from .llm import LLMBackend, ToolCall


@dataclass
class LeadAgentResult:
    """Result from the LeadAgent orchestrator."""

    content: str = ""
    success: bool = True
    rounds: int = 0
    agent_calls: list[dict[str, str]] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)


class LeadAgent:
    """Orchestrator agent that plans and delegates to sub-agents via tool-calling."""

    def __init__(
        self,
        agents: list[BaseAgent],
        llm: LLMBackend | None = None,
        model: str | None = None,
        human_in_the_loop: bool = False,
        max_rounds: int = 20,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.agents: dict[str, BaseAgent] = {a.name: a for a in agents}
        self._llm = llm or LLMBackend(model=model or "deepseek-chat")
        self.human_in_the_loop = human_in_the_loop
        self.max_rounds = max_rounds
        self._progress_callback = progress_callback

    def _emit_progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)

    def _build_system_prompt(self) -> str:
        agent_descriptions = "\n".join(
            f"- **{name}**: {agent.role[:200]}"
            for name, agent in self.agents.items()
        )
        return f"""You are the Lead Agent, the orchestrator of a multi-agent team working on vllm-omni and HPC tasks.

Your role:
- Analyze the user's task and create an execution plan
- Delegate work to specialized agents one at a time
- Review each agent's output and decide the next step
- You can loop agents (e.g., have coder implement changes suggested by optimizer, then verify)
- You can ask the user for clarification when needed
- When the task is complete, provide a comprehensive summary

Available agents:
{agent_descriptions}

Rules:
- ALWAYS delegate to a specific agent rather than trying to do work yourself
- Provide clear, specific task descriptions when delegating
- Include relevant context from previous agent outputs when delegating
- After receiving results, evaluate quality before deciding next steps
- If an agent's output is insufficient, re-delegate with more specific instructions
- Use ask_user only when you truly need user input
- Use finish when the task is fully complete with a summary of what was accomplished"""

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build OpenAI-format tool definitions for delegate_agent, ask_user, finish."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "delegate_agent",
                    "description": "Delegate a task to a sub-agent. The agent will execute the task and return results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {
                                "type": "string",
                                "enum": list(self.agents.keys()),
                                "description": "The name of the agent to delegate to.",
                            },
                            "task_description": {
                                "type": "string",
                                "description": "Clear description of what the agent should do, including any context from previous steps.",
                            },
                        },
                        "required": ["agent_name", "task_description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_user",
                    "description": "Ask the user a question for clarification or input.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask the user.",
                            },
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Signal that the task is complete. Provide a summary of what was accomplished.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Comprehensive summary of what was accomplished.",
                            },
                        },
                        "required": ["summary"],
                    },
                },
            },
        ]

    async def _get_user_input(self, question: str) -> str:
        """Get input from user via stdin."""
        print(f"\n[Lead Agent] {question}")
        try:
            return input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return "(user did not respond)"

    async def run(self, task: str, ctx: Context | None = None) -> LeadAgentResult:
        """Main entry: plan and execute task using sub-agents."""
        from .task import Task

        if ctx is None:
            ctx = Context(task=Task(description=task))

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": task},
        ]

        agent_calls: list[dict[str, str]] = []
        tools = self._build_tools()

        for round_num in range(self.max_rounds):
            self._emit_progress(f"[lead] round {round_num + 1}/{self.max_rounds}")
            response = await self._llm.complete(messages, tools=tools)
            self._llm._diagnose_output(response.content, "stop", agent_name="lead")

            if not response.tool_calls:
                # No tool call — model gave a final text response, treat as finish
                ctx.add_message("lead", response.content)
                return LeadAgentResult(
                    content=response.content,
                    success=True,
                    rounds=round_num + 1,
                    agent_calls=agent_calls,
                    artifacts=list(ctx.artifacts),
                )

            # Append assistant message with tool calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
            }
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                try:
                    args = json.loads(tc.arguments) if tc.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                tool_call_id = tc.id or f"call_{round_num}"

                if tc.function_name == "finish":
                    summary = args.get("summary", "")
                    self._emit_progress("[lead] finished")
                    ctx.add_message("lead", summary)
                    return LeadAgentResult(
                        content=summary,
                        success=True,
                        rounds=round_num + 1,
                        agent_calls=agent_calls,
                        artifacts=list(ctx.artifacts),
                    )

                elif tc.function_name == "ask_user":
                    question = args.get("question", "")
                    self._emit_progress(f"[lead] asking user: {question}")
                    if self.human_in_the_loop:
                        response_text = await self._get_user_input(question)
                    else:
                        response_text = "(auto-mode: skipping user question)"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": response_text,
                    })
                    ctx.add_message("lead", f"Asked user: {question} -> {response_text}")

                elif tc.function_name == "delegate_agent":
                    agent_name = args.get("agent_name", "")
                    sub_task = args.get("task_description", "")
                    sub_agent = self.agents.get(agent_name)

                    if sub_agent:
                        preview = " ".join(sub_task.split())
                        if len(preview) > 120:
                            preview = preview[:117] + "..."
                        self._emit_progress(f"[lead -> {agent_name}] {preview}")
                        result = await sub_agent.run(sub_task, ctx)
                        result_text = result.content
                        self._emit_progress(f"[{agent_name}] returned ({'ok' if result.success else 'incomplete'})")
                        agent_calls.append({
                            "agent_name": agent_name,
                            "task": sub_task,
                            "result_summary": result_text[:500],
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result_text,
                        })
                        ctx.add_message(
                            agent_name,
                            result_text,
                            tool_calls=[{
                                "id": tool_call_id,
                                "type": "function",
                                "function": {"name": tc.function_name, "arguments": tc.arguments},
                            }],
                        )
                    else:
                        error_msg = f"Error: Agent '{agent_name}' not found. Available: {list(self.agents.keys())}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": error_msg,
                        })
                        ctx.add_message("lead", error_msg)
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"Error: Unknown tool '{tc.function_name}'",
                    })

        return LeadAgentResult(
            content="Max rounds reached. Task may be incomplete.",
            success=False,
            rounds=self.max_rounds,
            agent_calls=agent_calls,
            artifacts=list(ctx.artifacts),
        )
