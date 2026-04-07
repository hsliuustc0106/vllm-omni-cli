"""Pipeline: orchestrate multiple agents in a DAG."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml

from .agent import BaseAgent
from .context import Context, AgentMessage
from .task import Task


class AgentStepResult:
    """Result from a single agent step in the pipeline."""

    def __init__(self, agent_name: str, content: str, success: bool = True) -> None:
        self.agent_name = agent_name
        self.content = content
        self.success = success


class PipelineResult:
    """Result from a full pipeline run."""

    def __init__(self) -> None:
        self.steps: list[AgentStepResult] = []
        self.success = True

    def add(self, step: AgentStepResult) -> None:
        self.steps.append(step)
        if not step.success:
            self.success = False


class Pipeline:
    """Execute agents in DAG order, passing Context between them."""

    def __init__(
        self,
        agents: list[BaseAgent] | None = None,
        edges: list[tuple[str, str]] | None = None,
    ) -> None:
        self.agents = agents or []
        self.edges = edges or []

    @staticmethod
    def from_yaml(path: Path) -> Pipeline:
        """Load pipeline definition from a YAML file."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        agents: list[BaseAgent] = []
        agent_map: dict[str, BaseAgent] = {}

        for adef in data.get("agents", []):
            agent = BaseAgent(name=adef["name"], role=adef.get("role", ""), model=adef.get("model", "gpt-4o"))
            agents.append(agent)
            agent_map[agent.name] = agent

        edges: list[tuple[str, str]] = []
        for e in data.get("edges", []):
            edges.append((e["from"], e["to"]))

        return Pipeline(agents=agents, edges=edges)

    def _topo_sort(self) -> list[str]:
        """Topological sort of agent names based on edges."""
        graph: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = defaultdict(int)
        names = [a.name for a in self.agents]
        for n in names:
            in_degree[n] = in_degree.get(n, 0)

        for src, dst in self.edges:
            graph[src].append(dst)
            in_degree[dst] += 1

        queue = deque(n for n in names if in_degree[n] == 0)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    async def run(self, task: str) -> PipelineResult:
        """Execute the pipeline on a task."""
        t = Task(description=task)
        ctx = Context(task=t)
        result = PipelineResult()

        agent_map = {a.name: a for a in self.agents}
        order = self._topo_sort()

        for name in order:
            agent = agent_map.get(name)
            if agent is None:
                result.add(AgentStepResult(name, f"Agent '{name}' not found.", success=False))
                continue
            step_result = await agent.run(task, ctx)
            ctx.add_message(name, step_result.content)
            ctx.add_artifact(name, "analysis", step_result.content)
            result.add(AgentStepResult(name, step_result.content, step_result.success))

            if not step_result.success:
                break

        return result
