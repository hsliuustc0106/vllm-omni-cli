"""Built-in tools."""

from .github import GitHubTool
from .shell import ShellTool
from .vllm import VllmTool

BUILTIN_TOOLS = {
    "github": GitHubTool,
    "shell": ShellTool,
    "vllm": VllmTool,
}

__all__ = ["GitHubTool", "ShellTool", "VllmTool", "BUILTIN_TOOLS"]
