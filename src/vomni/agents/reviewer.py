"""Reviewer agent for code review."""

from ..core.agent import BaseAgent


class ReviewerAgent(BaseAgent):
    """Code reviewer for vllm-omni."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(
            name="reviewer",
            role=(
                "You are a code reviewer for vllm-omni. You check code quality, correctness, "
                "performance implications, test coverage, documentation, and adherence to "
                "project conventions. You provide actionable feedback with specific suggestions."
            ),
            **kwargs,
        )
