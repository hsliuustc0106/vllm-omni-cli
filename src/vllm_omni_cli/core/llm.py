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
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


@dataclass
class LLMChunk:
    delta_content: str = ""
    delta_tool_calls: list[ToolCall] = field(default_factory=list)


class LLMBackend:
    """Unified LLM backend using litellm."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def _kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"model": self.model}
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.base_url:
            kw["api_base"] = self.base_url
        return kw

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
        return LLMResponse(content=msg.content or "", tool_calls=tool_calls, usage=usage, raw=raw)

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
            yield LLMChunk(delta_content=delta.content or "", delta_tool_calls=tc_list)
