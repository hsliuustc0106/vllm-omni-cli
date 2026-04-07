"""GitHub tool wrapping the gh CLI."""

from __future__ import annotations

from typing import Any

from ..core.tool import BaseTool


class GitHubTool(BaseTool):
    """Interact with GitHub via the gh CLI."""

    name = "github"
    description = "Interact with GitHub: manage PRs, issues, and reviews via the gh CLI."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_prs", "create_pr", "review_pr", "get_pr_diff", "add_comment", "list_issues"],
                "description": "The GitHub action to perform.",
            },
            "repo": {"type": "string", "description": "Repository in owner/repo format."},
            "args": {"type": "string", "description": "Additional arguments for the action."},
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs: Any) -> str:
        import asyncio

        action = kwargs.get("action", "")
        repo = kwargs.get("repo", "")
        args = kwargs.get("args", "")

        cmd_map = {
            "list_prs": f"gh pr list --repo {repo} {args}",
            "create_pr": f"gh pr create --repo {repo} {args}",
            "review_pr": f"gh pr review --repo {repo} {args}",
            "get_pr_diff": f"gh pr diff --repo {repo} {args}",
            "add_comment": f"gh pr comment --repo {repo} {args}",
            "list_issues": f"gh issue list --repo {repo} {args}",
        }

        cmd = cmd_map.get(action)
        if not cmd:
            return f"Unknown action: {action}. Valid: {', '.join(cmd_map.keys())}"

        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return f"Error: {stderr.decode()}"
        return stdout.decode()
