"""vllm tool wrapping vllm serve and bench commands."""

from __future__ import annotations

from typing import Any

from ..core.tool import BaseTool


class VllmTool(BaseTool):
    """Control vllm-omni serve and benchmark via CLI."""

    name = "vllm"
    description = "Manage vllm-omni: serve models, run benchmarks, stop instances."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["serve", "bench", "stop"],
                "description": "Action to perform: serve, bench, or stop.",
            },
            "model": {"type": "string", "description": "Model identifier (for serve/bench)."},
            "args": {"type": "string", "description": "Additional CLI arguments."},
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs: Any) -> str:
        import asyncio

        action = kwargs.get("action", "")
        model = kwargs.get("model", "")
        args = kwargs.get("args", "")

        if action == "serve":
            cmd = f"vllm serve {model} --omni {args}".strip()
        elif action == "bench":
            cmd = f"vllm bench serve --omni {args}".strip()
        elif action == "stop":
            cmd = "pkill -f 'vllm serve' || echo 'No vllm serve process found'"
        else:
            return f"Unknown action: {action}"

        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return f"Error: {stderr.decode()}"
        return stdout.decode()
