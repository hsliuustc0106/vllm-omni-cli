"""Coder agent for vllm-omni development."""

from ..core.react_agent import ReActAgent
from ..core.registry import register_agent


@register_agent("coder", scopes=["development", "inference"])
class CoderAgent(ReActAgent):
    """Expert Python/CUDA/PyTorch developer for model inference."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(
            name="coder",
            role=(
                "You are an expert Python/CUDA/PyTorch developer specializing in model "
                "inference. You write production-quality code for vllm-omni including model "
                "integration, custom pipelines, attention kernels, and distributed execution. "
                "You follow vllm-omni code conventions, write tests, and ensure type safety."
            ),
            **kwargs,
        )
