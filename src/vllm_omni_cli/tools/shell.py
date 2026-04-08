"""Shell tool for safe command execution."""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.tool import BaseTool

BLOCKED_COMMANDS = {"rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:"}
BROAD_SCAN_PATTERNS = (
    "find / ",
    "find /root",
    "find ~/",
    "find ~/.",
    "grep -R /",
    "grep -r /",
    "cat /root/.cache",
    "ls /root/.cache",
)


class ShellTool(BaseTool):
    """Execute shell commands safely with timeout support."""

    name = "shell"
    description = (
        "Execute a narrowly scoped shell command. Prefer repo-local inspection such as rg, ls, "
        "sed, git, or python -c on a specific file. Do not scan the whole filesystem or caches."
    )
    category = "execution"
    scopes = ["coder", "optimizer", "reviewer"]
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 60)."},
            "cwd": {"type": "string", "description": "Working directory."},
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", 60)
        cwd = kwargs.get("cwd")
        normalized = " ".join(command.strip().split())

        # Safety check
        for blocked in BLOCKED_COMMANDS:
            if blocked in command:
                return f"Blocked: command contains dangerous pattern '{blocked}'"

        for pattern in BROAD_SCAN_PATTERNS:
            if normalized.startswith(pattern) or pattern in normalized:
                return (
                    "Blocked: broad filesystem scans are not allowed. "
                    "Use a repo-scoped command with cwd set, or inspect a specific known path."
                )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            if proc.returncode != 0:
                return f"[exit {proc.returncode}]\n{stderr.decode()}\n{stdout.decode()}"
            return stdout.decode()
        except asyncio.TimeoutError:
            proc.kill()
            return f"Command timed out after {timeout}s"
