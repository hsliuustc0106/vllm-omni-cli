"""Task definition."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A unit of work for agents."""

    description: str
    constraints: list[str] = Field(default_factory=list)
    context: str = ""
