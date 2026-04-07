"""Shell tool for safe command execution."""

from __future__ import annotations

import shlex
import asyncio
from typing import Any

from ..core.tool import BaseTool

BLOCKED_COMMANDS = {"rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:"}


class ShellTool(BaseTool):
    """Execute shell commands safely with timeout support."""

    name = "shell"
    description = "Execute shell commands. Use 'run' for normal execution, 'run_with_timeout' for timeout."
    category = "execution"
    scopes = ["all"]
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

        # Safety check
        for blocked in BLOCKED_COMMANDS:
            if blocked in command:
                return f"Blocked: command contains dangerous pattern '{blocked}'"

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
