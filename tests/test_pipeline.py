"""Tests for Pipeline."""

import pytest
from pathlib import Path

from vllm_omni_cli.core.pipeline import Pipeline
from vllm_omni_cli.core.agent import BaseAgent


def _make_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_from_yaml_minimal(tmp_path: Path):
    yml = _make_yaml(tmp_path / "pipe.yaml", """
agents:
  - name: a1
    role: Agent 1
  - name: a2
    role: Agent 2
""")
    pipe = Pipeline.from_yaml(yml)
    assert len(pipe.agents) == 2
    assert pipe.agents[0].name == "a1"
    assert pipe.edges == []


def test_from_yaml_with_edges(tmp_path: Path):
    yml = _make_yaml(tmp_path / "pipe.yaml", """
agents:
  - name: a
  - name: b
edges:
  - from: a
    to: b
""")
    pipe = Pipeline.from_yaml(yml)
    assert pipe.edges == [("a", "b")]


def test_from_yaml_invalid(tmp_path: Path):
    yml = _make_yaml(tmp_path / "bad.yaml", "not: valid: yaml: [")
    # yaml.safe_load will raise; we just want it not to crash silently
    with pytest.raises(Exception):
        Pipeline.from_yaml(yml)


def test_topo_sort_linear(tmp_path: Path):
    yml = _make_yaml(tmp_path / "pipe.yaml", """
agents:
  - name: a
  - name: b
  - name: c
edges:
  - from: a
    to: b
  - from: b
    to: c
""")
    pipe = Pipeline.from_yaml(yml)
    order = pipe._topo_sort()
    assert order.index("a") < order.index("b") < order.index("c")


def test_topo_sort_cycle(tmp_path: Path):
    """Cycle in edges — topo sort will not include all nodes."""
    yml = _make_yaml(tmp_path / "cycle.yaml", """
agents:
  - name: x
  - name: y
edges:
  - from: x
    to: y
  - from: y
    to: x
""")
    pipe = Pipeline.from_yaml(yml)
    order = pipe._topo_sort()
    # With a cycle, not all nodes are reachable from zero in-degree
    assert len(order) < 2


def test_orphan_nodes_in_topo_sort(tmp_path: Path):
    """Nodes with no edges should still appear in topo sort."""
    yml = _make_yaml(tmp_path / "pipe.yaml", """
agents:
  - name: lone
  - name: connected1
  - name: connected2
edges:
  - from: connected1
    to: connected2
""")
    pipe = Pipeline.from_yaml(yml)
    order = pipe._topo_sort()
    assert "lone" in order
