"""Built-in agents for vllm-omni workflows."""

from .architect import ArchitectAgent
from .coder import CoderAgent
from .optimizer import OptimizerAgent
from .reviewer import ReviewerAgent

BUILTIN_AGENTS = {
    "architect": ArchitectAgent,
    "coder": CoderAgent,
    "optimizer": OptimizerAgent,
    "reviewer": ReviewerAgent,
}

__all__ = ["ArchitectAgent", "CoderAgent", "OptimizerAgent", "ReviewerAgent", "BUILTIN_AGENTS"]
