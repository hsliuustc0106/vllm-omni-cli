"""Architect agent for HPC and distributed inference design."""

from ..core.react_agent import ReActAgent
from ..core.registry import register_agent


@register_agent("architect", scopes=["hpc", "inference"])
class ArchitectAgent(ReActAgent):
    """Designs efficient inference pipelines for vllm-omni."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(
            name="architect",
            role=(
                "You are an HPC and distributed inference architect. You design efficient "
                "inference pipelines for vllm-omni, covering tensor parallelism, pipeline "
                "parallelism, HSDP, disaggregated serving, and stage configuration. You "
                "understand GPU memory hierarchies, NCCL communication patterns, and model "
                "sharding strategies. When designing solutions, consider throughput, latency, "
                "memory efficiency, and fault tolerance."
            ),
            **kwargs,
        )
