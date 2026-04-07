"""LLM backend via litellm."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCall:
    id: str = ""
    function_name: str = ""
    arguments: str = ""


@dataclass
class LLMResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


@dataclass
class LLMChunk:
    delta_content: str = ""
    delta_reasoning: str = ""
    delta_tool_calls: list[ToolCall] = field(default_factory=list)


class LLMBackend:
    """Unified LLM backend using litellm."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 8192, extra_body: dict | None = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

        # Token accumulation
        self._prompt_tokens = 0
        self._completion_tokens = 0

        # Display control for stream
        self.display_reasoning = True
        self.display_content = True

        # Thinking marker state
        self._in_think_tag = False

    def _kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"model": self.model}
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.base_url:
            kw["api_base"] = self.base_url
        kw["temperature"] = self.temperature
        kw["max_tokens"] = self.max_tokens
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        return kw

    @property
    def total_prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    def reset_token_stats(self):
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def _update_token_stats(self, usage: Usage | None):
        if usage:
            self._prompt_tokens += usage.prompt_tokens or 0
            self._completion_tokens += usage.completion_tokens or 0

    def _parse_response(self, raw: Any) -> LLMResponse:
        choice = raw.choices[0]
        msg = choice.message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id or "",
                        function_name=tc.function.name if tc.function else "",
                        arguments=tc.function.arguments if tc.function else "",
                    )
                )
        usage_info = raw.usage
        usage = Usage(
            prompt_tokens=getattr(usage_info, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_info, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_info, "total_tokens", 0) or 0,
        )
        reasoning = getattr(msg, 'reasoning_content', None) or ""
        return LLMResponse(content=msg.content or "", reasoning_content=reasoning,
                           tool_calls=tool_calls, usage=usage, raw=raw)

    def _diagnose_output(self, content: str, finish_reason: str, agent_name: str = "unknown") -> None:
        """Diagnose LLM output and log warnings for common issues."""
        if finish_reason == "length":
            logger.warning(f"[{agent_name}] Output truncated (finish_reason='length'), content may be incomplete")
        elif finish_reason == "content_filter":
            logger.warning(f"[{agent_name}] Output filtered by content safety")
        elif not content and finish_reason not in ("stop", "tool_calls", None):
            logger.warning(f"[{agent_name}] Empty output with finish_reason='{finish_reason}'")
        if not content and finish_reason == "stop":
            logger.debug(f"[{agent_name}] Empty content with finish_reason='stop'")

    @staticmethod
    def split_think(content: str) -> tuple[str, str]:
        """Split thinking content from response (DeepSeek style)."""
        for m in ["🤔", "\nReasoning:", "\n<think"]:
            pos = content.find(m)
            if pos != -1:
                reasoning = content[:pos].strip()
                answer = content[pos + len(m):].strip()
                return answer, reasoning
        return content, ""

    async def complete(
        self, messages: list[dict], tools: list[dict] | None = None, **kwargs: Any
    ) -> LLMResponse:
        kw = self._kwargs()
        kw["messages"] = messages
        if tools:
            kw["tools"] = tools
        kw.update(kwargs)
        raw = await litellm.acompletion(**kw)
        resp = self._parse_response(raw)
        self._update_token_stats(resp.usage)
        finish_reason = raw.choices[0].finish_reason if raw.choices else None
        self._diagnose_output(resp.content, finish_reason or "stop", agent_name="unknown")
        return resp

    async def stream(
        self, messages: list[dict], tools: list[dict] | None = None, **kwargs: Any
    ) -> AsyncIterator[LLMChunk]:
        kw = self._kwargs()
        kw["messages"] = messages
        kw["stream"] = True
        if tools:
            kw["tools"] = tools
        kw.update(kwargs)
        async for chunk in await litellm.acompletion(**kw):
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            tc_list: list[ToolCall] = []
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_list.append(
                        ToolCall(
                            id=tc.id or "",
                            function_name=tc.function.name if tc.function else "",
                            arguments=tc.function.arguments if tc.function else "",
                        )
                    )
            # Extract reasoning_content
            reasoning_delta = getattr(delta, 'reasoning_content', None) or ""
            content_delta = delta.content or ""

            if reasoning_delta and self.display_reasoning:
                yield LLMChunk(delta_reasoning=reasoning_delta, delta_content="")
            if content_delta and self.display_content:
                yield LLMChunk(delta_content=content_delta, delta_reasoning="")
            if tc_list:
                yield LLMChunk(delta_tool_calls=tc_list)
