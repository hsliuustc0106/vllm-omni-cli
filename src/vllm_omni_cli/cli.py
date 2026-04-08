"""CLI entry point for vllm-omni-cli."""

from __future__ import annotations

import asyncio
import logging
import os
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
from .core.request import ChatSessionState, load_installed_skill_registry, prepare_agent_request, render_prepared_task
from .core.tool import ToolRegistry
from .core.skill import SkillAdapter, SkillRegistry
from .model_catalog import load_model_alias_entries, resolve_model_alias
from .recipes_sync import sync_recipes_catalog
from .tools import BUILTIN_TOOLS

app = typer.Typer(name="vllm_omni_cli", help="vllm-omni-cli: Multi-agent framework for vllm-omni")
console = Console()


def _configure_logging(debug: bool = False) -> None:
    """Initialize CLI logging once so agent loop diagnostics are visible."""
    env_debug = os.environ.get("VLLM_OMNI_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    level = logging.DEBUG if debug or env_debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _print_request_debug_summary(request) -> None:
    table = Table(title="Request Routing")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("request_id", request.request_id)
    table.add_row("mode", request.mode)
    table.add_row("family", request.resolved_model_family or "(none)")
    table.add_row("categories", ", ".join(request.task_categories) if request.task_categories else "(none)")
    table.add_row("hardware_hints", ", ".join(request.hardware_hints) if request.hardware_hints else "(none)")
    table.add_row("agents", ", ".join(request.target_agents) if request.target_agents else "(none)")
    table.add_row("skills", ", ".join(request.merged_skill_refs) if request.merged_skill_refs else "(none)")
    table.add_row("notes", " | ".join(request.routing_notes) if request.routing_notes else "(none)")
    console.print(table)


def _progress_printer(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


@app.command()
def run(
    task: str = typer.Argument(help="Task description"),
    agents: Optional[str] = typer.Option(None, help="Comma-separated agent names"),
    skills: Optional[str] = typer.Option(None, help="Comma-separated skill paths"),
    pipeline: Optional[Path] = typer.Option(None, help="Pipeline YAML file"),
    model: Optional[str] = typer.Option(None, help="LLM model identifier"),
    human_in_the_loop: bool = typer.Option(False, "--human-in-the-loop", help="Enable human review"),
    debug: bool = typer.Option(False, "--debug", help="Enable verbose agent loop logging"),
    quick: bool = typer.Option(False, "--quick", help="Return a fast single-agent draft without orchestration"),
) -> None:
    """Run a task with specified agents."""
    _configure_logging(debug=debug)
    cfg = config_list()
    factory = LLMFactory(config=cfg.get("llm", {}))
    skill_registry = load_installed_skill_registry()

    # If --model is specified, override all levels
    if model:
        lead_llm = factory.create(model=model)
    else:
        lead_llm = factory.create(model_level="standard")

    tool_reg = ToolRegistry()
    tool_reg.discover()
    for ToolCls in BUILTIN_TOOLS.values():
        tool_reg.register(ToolCls())

    manual_skill_refs = _parse_skill_refs(skills)
    requested_agents = _parse_agent_names(agents)
    request = prepare_agent_request(
        task,
        "quick" if quick else "orchestrated",
        explicit_model=model,
        manual_skill_refs=manual_skill_refs,
        target_agents=requested_agents,
        debug=debug,
        human_in_the_loop=human_in_the_loop,
        skill_registry=skill_registry,
    )
    prepared_task = render_prepared_task(request)
    if debug:
        _print_request_debug_summary(request)

    skill_objs = _load_skills_from_refs(request.merged_skill_refs, skill_registry)
    agent_list = _resolve_agents(request.target_agents)
    agent_instances = []
    for cls in agent_list:
        if model:
            agent_llm = factory.create(model=model)
        else:
            agent_llm = factory.create_for_agent(cls.name)
        agent_instances.append(
            cls(
                tools=list(tool_reg._tools.values()),
                skills=skill_objs,
                llm=agent_llm,
                progress_callback=_progress_printer,
            )
        )

    if pipeline and pipeline.exists():
        # Pipeline YAML loads agents in order, but LeadAgent decides execution
        pipe = Pipeline.from_yaml(pipeline)
        agent_instances = pipe.agents or agent_instances

    if not agent_instances:
        console.print("[red]No agents available. Check --agents or pipeline config.[/red]")
        raise typer.Exit(1)

    if quick:
        quick_agent = agent_instances[0]
        console.print(f"[green]Quick mode using {quick_agent.name}[/green]")
        if hasattr(quick_agent, "run_single_turn"):
            result = asyncio.run(quick_agent.run_single_turn(prepared_task))
        else:
            result = asyncio.run(quick_agent.run(prepared_task, Context(task=Task(description=prepared_task))))
        console.print("\n[bold]Quick Result:[/bold]")
        console.print(result.content)
        return

    lead = LeadAgent(
        agents=agent_instances,
        llm=lead_llm,
        human_in_the_loop=human_in_the_loop,
        progress_callback=_progress_printer,
    )
    console.print(f"[green]Lead Agent orchestrating {len(agent_instances)} agents: {', '.join(a.name for a in agent_instances)}[/green]")
    result = asyncio.run(lead.run(prepared_task))
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
    skills: Optional[str] = typer.Option(None, help="Comma-separated skill names or paths"),
    debug: bool = typer.Option(False, "--debug", help="Enable verbose request preparation logging"),
) -> None:
    """Interactive chat with an agent."""
    _configure_logging(debug=debug)
    cls = BUILTIN_AGENTS.get(agent_name)
    if not cls:
        console.print(f"[red]Agent '{agent_name}' not found[/red]")
        raise typer.Exit(1)

    cfg = config_list()
    factory = LLMFactory(config=cfg.get("llm", {}))
    skill_registry = load_installed_skill_registry()
    agent_llm = factory.create(model=model) if model else factory.create_for_agent(agent_name)
    agent = cls(llm=agent_llm, skills=_load_skills_from_refs(_parse_skill_refs(skills), skill_registry))
    history: list[dict] = []
    chat_session = ChatSessionState()

    console.print(f"[green]Chatting with {agent.name}. Ctrl+C to exit.[/green]")
    try:
        while True:
            message = typer.prompt("You")
            if not message.strip():
                continue
            request = prepare_agent_request(
                message,
                "chat",
                explicit_model=model,
                manual_skill_refs=_parse_skill_refs(skills),
                target_agents=[agent_name],
                debug=debug,
                skill_registry=skill_registry,
                chat_session=chat_session,
            )
            if debug:
                _print_request_debug_summary(request)
            agent.skills = _load_skills_from_refs(request.merged_skill_refs, skill_registry)
            prepared_message = render_prepared_task(request)
            reply = asyncio.run(agent.chat(prepared_message, history))
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


@app.command(name="list")
def list_alias(
    kind: str = typer.Argument(help="What to list: agents, skills, or tools"),
) -> None:
    """Alias for list-items."""
    list_items(kind)


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


@app.command()
def catalog(
    action: str = typer.Argument(help="sync-recipes|list|resolve"),
    alias: Optional[str] = typer.Argument(None, help="Alias to resolve (for resolve)"),
) -> None:
    """Manage the local model alias catalog."""
    if action == "sync-recipes":
        result = sync_recipes_catalog()
        console.print(
            f"[green]Synced recipes catalog:[/green] imported={result['imported_count']} "
            f"preserved={result['preserved_count']} path={result['output_path']}"
        )
    elif action == "list":
        table = Table(title="Model Alias Catalog")
        table.add_column("Alias")
        table.add_column("Status")
        table.add_column("Family")
        table.add_column("Source")
        for entry in load_model_alias_entries():
            table.add_row(entry.alias, entry.status, entry.canonical_family, entry.source)
        console.print(table)
    elif action == "resolve":
        if not alias:
            console.print("[red]Usage: vllm_omni_cli catalog resolve <alias>[/red]")
            raise typer.Exit(1)
        entry = resolve_model_alias(alias)
        if entry is None:
            console.print(f"[yellow]Alias '{alias}' not found[/yellow]")
            raise typer.Exit(1)
        console.print(f"alias: {entry.alias}")
        console.print(f"status: {entry.status}")
        console.print(f"family: {entry.canonical_family}")
        console.print(f"note: {entry.note}")
        console.print(f"source: {entry.source}")
        console.print(f"suggestions: {', '.join(entry.suggestions) if entry.suggestions else '(none)'}")
    else:
        console.print("[red]Usage: vllm_omni_cli catalog sync-recipes | vllm_omni_cli catalog list | vllm_omni_cli catalog resolve <alias>[/red]")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(version: bool = typer.Option(False, "--version", "-v", help="Show version")) -> None:
    if version:
        console.print(f"vo v{__version__}")


def _resolve_agents(agent_names: list[str]) -> list[type[BaseAgent]]:
    if agent_names:
        classes = []
        for name in agent_names:
            cls = BUILTIN_AGENTS.get(name)
            if cls:
                classes.append(cls)
            else:
                console.print(f"[yellow]Agent '{name}' not found, skipping[/yellow]")
        return classes
    return list(BUILTIN_AGENTS.values())


def _parse_agent_names(agents_str: Optional[str]) -> list[str]:
    if not agents_str:
        return []
    return [name.strip() for name in agents_str.split(",") if name.strip()]


def _parse_skill_refs(skills_str: Optional[str]) -> list[str]:
    if not skills_str:
        return []
    return [ref.strip() for ref in skills_str.split(",") if ref.strip()]


def _load_skills_from_refs(skill_refs: list[str], registry: SkillRegistry) -> list:
    skills = []
    loaded_names: set[str] = set()
    for ref in skill_refs:
        path = Path(ref)
        if path.exists():
            if path.is_dir() and (path / "SKILL.md").exists():
                loaded = [SkillAdapter.load_from_directory(path)]
            elif path.is_dir():
                loaded = SkillAdapter.load_from_repo(path)
            else:
                loaded = []
        else:
            meta = registry.get(ref)
            if meta is None:
                console.print(f"[yellow]Skill '{ref}' not found, skipping[/yellow]")
                continue
            loaded = [SkillAdapter.load_from_directory(Path(meta.source_path))]

        for skill in loaded:
            if skill.name not in loaded_names:
                loaded_names.add(skill.name)
                skills.append(skill)
    return skills
