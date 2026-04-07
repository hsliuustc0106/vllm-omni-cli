"""Tests for Jinja2 prompt templates."""

import pytest

from vllm_omni_cli.core.prompts import PromptTemplate, PromptLoader


class TestPromptTemplate:
    def test_prompt_template_format(self):
        tmpl = PromptTemplate("Hello, {{ name }}!")
        result = tmpl.format(name="World")
        assert result == "Hello, World!"

    def test_prompt_template_with_variables(self):
        tmpl = PromptTemplate("Agent: {{ agent_name }}\nRole: {{ role }}\nTask: {{ task }}")
        result = tmpl.format(agent_name="coder", role="developer", task="write code")
        assert "coder" in result
        assert "developer" in result
        assert "write code" in result

    def test_prompt_template_no_variables(self):
        tmpl = PromptTemplate("Static prompt.")
        assert tmpl.format() == "Static prompt."


class TestPromptLoader:
    def test_prompt_loader_from_string(self):
        loader = PromptLoader()
        tmpl = loader.from_string("Hello, {{ who }}!")
        assert tmpl.format(who="Alice") == "Hello, Alice!"
