"""Execution context passed between agents."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .task import Task


class Artifact(BaseModel):
    """An output artifact produced by an agent."""

    name: str
    type: str = Field(description='e.g. "code", "config", "file", "analysis"')
    content: str


class AgentMessage(BaseModel):
    """A message logged by an agent during execution."""

    agent_name: str
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Context(BaseModel):
    """Shared execution context for a pipeline run."""

    task: Task
    state: dict[str, Any] = Field(default_factory=dict)
    messages: list[AgentMessage] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)

    def add_message(self, agent_name: str, content: str, tool_calls: list[dict] | None = None) -> None:
        self.messages.append(
            AgentMessage(agent_name=agent_name, content=content, tool_calls=tool_calls or [])
        )

    def add_artifact(self, name: str, type_: str, content: str) -> None:
        self.artifacts.append(Artifact(name=name, type=type_, content=content))
