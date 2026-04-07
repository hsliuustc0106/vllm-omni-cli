"""Jinja2 prompt template support."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, BaseLoader, FileSystemLoader


class PromptTemplate:
    """Jinja2-based prompt template."""

    def __init__(self, template_str: str):
        self._env = Environment(loader=BaseLoader())
        self._template = self._env.from_string(template_str)

    def format(self, **kwargs) -> str:
        return self._template.render(**kwargs)


class PromptLoader:
    """Load Jinja2 prompt templates from directories."""

    def __init__(self, search_paths: list[str | Path] | None = None):
        paths = [str(p) for p in (search_paths or [])]
        self._env = Environment(
            loader=FileSystemLoader(paths) if paths else BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def load(self, template_name: str) -> PromptTemplate:
        template = self._env.get_template(template_name)

        class _TemplateWrapper:
            def __init__(self, tmpl):
                self._tmpl = tmpl

            def format(self, **kwargs) -> str:
                return self._tmpl.render(**kwargs)

        return _TemplateWrapper(template)  # type: ignore

    def from_string(self, template_str: str) -> PromptTemplate:
        return PromptTemplate(template_str)
