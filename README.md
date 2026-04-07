# vllm-omni-cli (`vo`)

Multi-agent collaboration framework for [vllm-omni](https://github.com/vllm-project/vllm-omni) and HPC inference scenarios.

## Features

- **Composable Agents & Skills** — Mix and match agents, skills, and tools freely
- **Plugin System** — Extend via `entry_points` (like pytest plugins)
- **Skills Bridge** — Load vllm-omni-skills (SKILL.md format) directly
- **Unified LLM Backend** — Use any LLM via litellm (OpenAI, Anthropic, local, etc.)
- **Pipeline DAG** — Orchestrate multi-agent workflows with conditional routing
- **Built-in Agents** — Architect, Coder, Optimizer, Reviewer
- **Built-in Tools** — GitHub, vllm-omni CLI, Shell

## Install

```bash
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

### Quick Configuration

Set your LLM backend via environment variables:

```bash
# DeepSeek (default)
export VLLM_OMNI_AGENTS_BASE_URL="https://api.deepseek.com/beta/"
export VLLM_OMNI_AGENTS_API_KEY="your-key-here"
export VLLM_OMNI_AGENTS_MODEL_NAME="deepseek-chat"

# Or use any OpenAI-compatible API
export VLLM_OMNI_AGENTS_BASE_URL="https://your-api-endpoint/v1"
export VLLM_OMNI_AGENTS_API_KEY="your-key-here"
export VLLM_OMNI_AGENTS_MODEL_NAME="your-model-name"
```

Then verify:

```bash
vo config list
```

## Quick Start

```bash
# Initialize config
vo config init

# Set your LLM model
vo config set llm.model claude-3-5-sonnet-20241022
vo config set llm.api_key sk-...

# Run a task with all agents
vo run "Design a distributed serving pipeline for Llama-3-70B"

# Use specific agents
vo run "Optimize attention kernel" --agents optimizer,coder

# Interactive chat
vo chat --agent architect

# List available resources
vo list agents
vo list tools
vo list skills
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CLI (typer)                       │
│                  vo run / chat / config              │
├─────────────────────────────────────────────────────┤
│                   Pipeline (DAG)                     │
│         orchestrates agents in topo order            │
├──────────┬──────────┬──────────┬────────────────────┤
│Architect │  Coder   │Optimizer │ Reviewer  │ Custom │
│  Agent   │  Agent   │  Agent   │  Agent    │ Agents │
├──────────┴──────────┴──────────┴────────────────────┤
│              Skills + Knowledge Base                  │
│         (loaded from SKILL.md directories)           │
├──────────┬──────────┬──────────┬────────────────────┤
│ GitHub   │  vllm    │  Shell   │  Custom            │
│  Tool    │  Tool    │  Tool    │  Tools             │
├──────────┴──────────┴──────────┴────────────────────┤
│              LLM Backend (litellm)                    │
│       OpenAI · Anthropic · Local · Any provider      │
└─────────────────────────────────────────────────────┘
```

## Pipeline Definition

Create a `pipeline.yaml`:

```yaml
agents:
  - name: architect
    role: "You are an HPC architect."
    model: gpt-4o
  - name: coder
    role: "You are a Python developer."
    model: gpt-4o
  - name: reviewer
    role: "You are a code reviewer."
    model: gpt-4o

edges:
  - from: architect
    to: coder
  - from: coder
    to: reviewer
```

Run it:

```bash
vo run "Build a custom attention kernel" --pipeline pipeline.yaml
```

## Writing Plugins

### Custom Tool

```python
# my_tool.py
from vomni.core.tool import BaseTool

class SlackTool(BaseTool):
    name = "slack"
    description = "Send messages to Slack"
    parameters = {
        "type": "object",
        "properties": {
            "channel": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["channel", "message"],
    }

    async def execute(self, **kwargs):
        # Your implementation
        return f"Sent to {kwargs['channel']}"
```

Register in `pyproject.toml`:

```toml
[project.entry-points."vomni.tools"]
slack = "my_tool:SlackTool"
```

### Custom Agent

```python
# my_agent.py
from vomni.core.agent import BaseAgent

class DataAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="data-scientist",
            role="You analyze ML training data and metrics.",
            **kwargs,
        )
```

Register:

```toml
[project.entry-points."vomni.agents"]
data-scientist = "my_agent:DataAgent"
```

### Custom Skill (SKILL.md format)

Create a directory with `SKILL.md`:

```
my-skill/
└── SKILL.md
```

```markdown
---
name: cuda-tuning
description: CUDA kernel optimization patterns
tools: [shell, vllm]
---

# CUDA Tuning

## Patterns
- Use shared memory for reduction
- Prefer coalesced memory access
- Use warp-level primitives (shfl_sync)
```

Install:

```bash
vo skill install /path/to/my-skill
```

## Configuration

Config lives at `~/.vo/config.toml`:

```toml
[llm]
model = "gpt-4o"
api_key = ""
base_url = ""

[tools]
github_token = ""

[skills]
paths = []

[defaults]
agents = ["architect", "coder", "reviewer"]
human_in_the_loop = false
```

## Development

```bash
pip install -e ".[dev]"
ruff check src/
pytest tests/
```

## Links

- [vllm-omni](https://github.com/vllm-project/vllm-omni) — Multi-modal inference engine
- [vllm-omni-skills](https://github.com/vllm-project/vllm-omni-skills) — Agent skill library

## License

Apache 2.0
