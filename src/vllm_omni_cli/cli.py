"""CLI entry point for vllm-omni-cli."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .agents import BUILTIN_AGENTS
from .config import config_get, config_init, config_list, config_set
from .core.agent import BaseAgent
from .core.context import Context, Task
from .core.llm import LLMBackend
from .core.llm_factory import LLMFactory
from .core.lead_agent import LeadAgent
from .core.pipeline import Pipeline
from .core.tool import ToolRegistry
from .core.skill import SkillAdapter
from .tools import BUILTIN_TOOLS

app = typer.Typer(name="vllm_omni_cli", help="vllm-omni-cli: Multi-agent framework for vllm-omni")
console = Console()


@app.command()
def run(
    task: str = typer.Argument(help="Task description"),
    agents: Optional[str] = typer.Option(None, help="Comma-separated agent names"),
    skills: Optional[str] = typer.Option(None, help="Comma-separated skill paths"),
    pipeline: Optional[Path] = typer.Option(None, help="Pipeline YAML file"),
    model: Optional[str] = typer.Option(None, help="LLM model identifier"),
    human_in_the_loop: bool = typer.Option(False, "--human-in-the-loop", help="Enable human review"),
) -> None:
    """Run a task with specified agents."""
    cfg = config_list()
    factory = LLMFactory(config=cfg.get("llm", {}))

    # If --model is specified, override all levels
    if model:
        lead_llm = factory.create(model=model)
    else:
        lead_llm = factory.create(model_level="standard")

    tool_reg = ToolRegistry()
    tool_reg.discover()
    for ToolCls in BUILTIN_TOOLS.values():
        tool_reg.register(ToolCls())

    skill_objs = _load_skills(skills)
    agent_list = _resolve_agents(agents, model)
    agent_instances = []
    for cls in agent_list:
        if model:
            agent_llm = factory.create(model=model)
        else:
            agent_llm = factory.create_for_agent(cls.name)
        agent_instances.append(cls(tools=list(tool_reg._tools.values()), skills=skill_objs, llm=agent_llm))

    if pipeline and pipeline.exists():
        # Pipeline YAML loads agents in order, but LeadAgent decides execution
        pipe = Pipeline.from_yaml(pipeline)
        agent_instances = pipe.agents or agent_instances

    if not agent_instances:
        console.print("[red]No agents available. Check --agents or pipeline config.[/red]")
        raise typer.Exit(1)

    lead = LeadAgent(
        agents=agent_instances,
        llm=lead_llm,
        human_in_the_loop=human_in_the_loop,
    )
    console.print(f"[green]Lead Agent orchestrating {len(agent_instances)} agents: {', '.join(a.name for a in agent_instances)}[/green]")
    result = asyncio.run(lead.run(task))
    console.print(f"\n[bold]Result ({result.rounds} rounds):[/bold]")
    console.print(result.content)
    if result.agent_calls:
        console.print(f"\n[dim]Agent calls: {len(result.agent_calls)}[/dim]")
        for call in result.agent_calls:
            console.print(f"  [cyan]{call['agent_name']}:[/cyan] {call['task'][:80]}")


@app.command()
def chat(
    agent_name: str = typer.Option("coder", "--agent", help="Agent to chat with"),
    model: Optional[str] = typer.Option(None, help="LLM model"),
) -> None:
    """Interactive chat with an agent."""
    cls = BUILTIN_AGENTS.get(agent_name)
    if not cls:
        console.print(f"[red]Agent '{agent_name}' not found[/red]")
        raise typer.Exit(1)

    cfg = config_list()
    factory = LLMFactory(config=cfg.get("llm", {}))
    agent_llm = factory.create(model=model) if model else factory.create_for_agent(agent_name)
    agent = cls(llm=agent_llm)
    history: list[dict] = []

    console.print(f"[green]Chatting with {agent.name}. Ctrl+C to exit.[/green]")
    try:
        while True:
            message = typer.prompt("You")
            if not message.strip():
                continue
            reply = asyncio.run(agent.chat(message, history))
            console.print(f"[bold]{agent.name}:[/bold] {reply}")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Bye![/dim]")


@app.command()
def list_items(
    kind: str = typer.Argument(help="What to list: agents, skills, or tools"),
) -> None:
    """List available agents, skills, or tools."""
    if kind == "agents":
        table = Table(title="Agents")
        table.add_column("Name")
        table.add_column("Source")
        for name, cls in BUILTIN_AGENTS.items():
            table.add_row(name, "builtin")
        console.print(table)
    elif kind == "skills":
        paths_str = config_get("skills.paths")
        if paths_str:
            import ast
            paths = ast.literal_eval(paths_str)
            table = Table(title="Skills")
            table.add_column("Name")
            table.add_column("Path")
            for p in paths:
                p_path = Path(p)
                if p_path.exists():
                    skills = SkillAdapter.load_from_repo(p_path)
                    for s in skills:
                        table.add_row(s.name, str(p_path / s.name))
                else:
                    table.add_row("—", f"{p} (not found)")
            console.print(table)
        else:
            console.print("[dim]No skill paths configured. Use 'vo config set skills.paths'[/dim]")
    elif kind == "tools":
        table = Table(title="Tools")
        table.add_column("Name")
        table.add_column("Description")
        for name, cls in BUILTIN_TOOLS.items():
            table.add_row(name, cls().description[:60])
        console.print(table)
    else:
        console.print(f"[red]Unknown kind: {kind}. Use agents|skills|tools[/red]")
        raise typer.Exit(1)


@app.command()
def config(
    action: str = typer.Argument(help="init|set|get|list"),
    key: Optional[str] = typer.Argument(None, help="Config key (for set/get)"),
    value: Optional[str] = typer.Argument(None, help="Config value (for set)"),
) -> None:
    """Manage configuration."""
    if action == "init":
        cfg = config_init()
        console.print(f"[green]Config initialized at ~/.vo/config.toml[/green]")
    elif action == "set":
        if not key or not value:
            console.print("[red]Usage: vo config set <key> <value>[/red]")
            raise typer.Exit(1)
        config_set(key, value)
        console.print(f"[green]Set {key} = {value}[/green]")
    elif action == "get":
        if not key:
            console.print("[red]Usage: vo config get <key>[/red]")
            raise typer.Exit(1)
        console.print(config_get(key))
    elif action == "list":
        import pprint
        console.print(pprint.pformat(config_list()))
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


@app.command()
def skill(
    action: str = typer.Argument(help="install|list|search"),
    path: Optional[str] = typer.Argument(None, help="Path (for install)"),
) -> None:
    """Manage skills."""
    if action == "list":
        list_items("skills")
    elif action == "install" and path:
        import ast
        current = config_get("skills.paths")
        paths = ast.literal_eval(current) if current else []
        if path not in paths:
            paths.append(path)
            config_set("skills.paths", str(paths))
        console.print(f"[green]Added skill path: {path}[/green]")
    else:
        console.print("[red]Usage: vo skill install <path> | vo skill list[/red]")


@app.command()
def tool(
    action: str = typer.Argument(help="add|list"),
) -> None:
    """Manage tools."""
    if action == "list":
        list_items("tools")
    else:
        console.print("[dim]Tool discovery happens automatically via entry_points.[/dim]")


@app.callback(invoke_without_command=True)
def main(version: bool = typer.Option(False, "--version", "-v", help="Show version")) -> None:
    if version:
        console.print(f"vo v{__version__}")


def _resolve_agents(agents_str: Optional[str], model: Optional[str]) -> list[type[BaseAgent]]:
    if agents_str:
        names = [n.strip() for n in agents_str.split(",")]
        classes = []
        for name in names:
            cls = BUILTIN_AGENTS.get(name)
            if cls:
                classes.append(cls)
            else:
                console.print(f"[yellow]Agent '{name}' not found, skipping[/yellow]")
        return classes
    return list(BUILTIN_AGENTS.values())


def _load_skills(skills_str: Optional[str]):
    if not skills_str:
        return []
    skills = []
    for p in skills_str.split(","):
        path = Path(p.strip())
        if path.exists():
            skills.extend(SkillAdapter.load_from_repo(path))
    return skills
