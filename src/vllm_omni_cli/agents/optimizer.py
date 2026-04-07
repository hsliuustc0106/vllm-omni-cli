"""Optimizer agent for inference performance."""

from ..core.agent import BaseAgent


class OptimizerAgent(BaseAgent):
    """Performance optimization expert for deep learning inference."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(
            name="optimizer",
            role=(
                "You are a performance optimization expert for deep learning inference. You "
                "analyze profiling data (nsight, torch profiler), identify bottlenecks in "
                "compute, memory, and communication, and recommend optimizations including "
                "quantization, kernel fusion, cache strategies, and scheduling improvements."
            ),
            **kwargs,
        )
