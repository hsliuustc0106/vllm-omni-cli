"""Built-in tools."""

from .github import GitHubTool
from .model_resolver import ModelResolverTool
from .shell import ShellTool
from .vllm import VllmTool

BUILTIN_TOOLS = {
    "github": GitHubTool,
    "model_resolver": ModelResolverTool,
    "shell": ShellTool,
    "vllm": VllmTool,
}

__all__ = ["GitHubTool", "ModelResolverTool", "ShellTool", "VllmTool", "BUILTIN_TOOLS"]
