# vllm-omni-cli

面向 vllm-omni 与高性能计算场景的 LLM 多 Agent 协作框架。通过智能 Agent 协同，将高性能代码的开发与优化效率提升到新的高度。

- **可组合性**：Agent、Skill、Tools 可灵活组合，适应不同任务需求
- **可扩展性**：通过 Skill 系统和 Agent 注册机制，社区可轻松扩展能力
- **Skills 桥接**：直接加载 [vllm-omni-skills](https://github.com/hsliuustc0106/vllm-omni-skills)（SKILL.md 格式）
- **统一 LLM 后端**：通过 litellm 接入任意 LLM（glm-5.1 (智谱 GLM)、OpenAI、Anthropic、本地模型等）
- **Lead Agent**：智能编排器动态规划 Agent 执行顺序，支持循环、条件路由和人工介入
- **插件系统**：通过 `entry_points` 扩展 Tools/Skills/Agents（类似 pytest 插件机制）

## 安装

```bash
# 1. 创建环境
conda create -n vllm_omni_agents python=3.11
conda activate vllm_omni_agents

# 2. 克隆 & 安装
git clone https://github.com/hsliuustc0106/vllm-omni-cli.git
cd vllm-omni-cli
pip install -e .
```

## 快速配置

通过环境变量配置 LLM 后端（环境变量优先级高于配置文件）：

```bash
# glm-5.1 (智谱 GLM)（推荐）
export VLLM_OMNI_AGENTS_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export VLLM_OMNI_AGENTS_API_KEY="your-api-key"
export VLLM_OMNI_AGENTS_MODEL_NAME="glm-5.1"

# 或使用任意 OpenAI 兼容 API
export VLLM_OMNI_AGENTS_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export VLLM_OMNI_AGENTS_API_KEY="your-api-key"
export VLLM_OMNI_AGENTS_MODEL_NAME="glm-5.1"
```

验证配置：

```bash
vllm_omni_cli config list
```

## 快速开始

```bash
# 初始化配置
vllm_omni_cli config init

# 运行任务（自动编排 Agent）
vllm_omni_cli run "Design a distributed serving pipeline for Qwen3-Omni"

# 查看多 Agent 进度与路由信息
vllm_omni_cli run --debug "Design the least latency deployment strategy for qwen-image on 2x NVIDIA L20"

# 快速草案模式（适合 CLI 早期阶段，快速拿到可用答案）
vllm_omni_cli run --quick --agents architect "Design the least latency deployment strategy for qwen-image with input 1024*1024 in 2x NVIDIA L20, each 46068 MiB VRAM, driver 570.211.01. GPU-to-GPU topology is NODE, not NVLink. Both GPUs are attached to NUMA node 1, CPU affinity 20-39,60-79. Host CPU is Intel Xeon Gold 6133, 2 sockets / 80 logical CPUs."

# 指定 Agent
vllm_omni_cli run "Optimize attention kernel for NPU" --agents optimizer,coder

# 使用 Pipeline YAML 定义工作流
vllm_omni_cli run "Build a custom attention kernel" --pipeline pipeline.yaml

# 交互式对话
vllm_omni_cli chat --agent architect --skills vllm-omni-image-gen,vllm-omni-hardware

# 查看可用资源
vllm_omni_cli list agents
vllm_omni_cli list tools
vllm_omni_cli list skills

# 同步 vllm-project/recipes 中的模型配方别名到本地目录 (~/.vo/model_aliases.json)
vllm_omni_cli catalog sync-recipes

# 查看某个模型别名的解析结果
vllm_omni_cli catalog resolve qwen-image
```

这个示例适合验证 Agent 是否会结合真实硬件拓扑给出低延迟部署建议，例如：
- 单卡优先还是 `TP=2`
- 是否应避免跨 GPU 同步
- NUMA 绑定与 CPU 亲和性设置
- 第二张 GPU 更适合做副本还是做张量并行

如果模型名本身不够精确（例如 `qwen-image`），可以先同步 recipes 配方目录，再让 CLI 基于本地别名目录做更稳健的解析。

调试模式下会输出一份简短的路由摘要，并显示类似 `[lead -> architect]`、`[optimizer] tool -> shell` 这样的进度行，方便观察多 Agent 的协作过程。

## 架构

```
┌──────────────────────────────────────────────────────┐
│                      CLI (typer)                      │
│            vllm_omni_cli run / chat / config              │
├──────────────────────────────────────────────────────┤
│                    Lead Agent (编排器)                   │
│         动态规划 Agent 执行顺序，支持循环和条件路由          │
├──────────┬──────────┬──────────┬─────────────────────┤
│Architect │  Coder   │Optimizer │ Reviewer │  Custom   │
│  Agent   │  Agent   │  Agent   │  Agent   │  Agents  │
├──────────┴──────────┴──────────┴─────────────────────┤
│               Skills + Knowledge Base                 │
│          （从 SKILL.md 目录自动加载）                   │
├──────────┬──────────┬──────────┬─────────────────────┤
│ GitHub   │  vllm    │  Shell   │  Custom              │
│  Tool    │  Tool    │  Tool    │  Tools               │
├──────────┴──────────┴──────────┴─────────────────────┤
│              LLM Backend (litellm)                     │
│       glm-5.1 (智谱 GLM) · OpenAI · Anthropic · 本地 · 任意      │
└──────────────────────────────────────────────────────┘
```

### 内置 Agent

| Agent | 职责 |
|-------|------|
| **Architect** | HPC/分布式推理架构设计，精通 TP/PP/HSDP/Disaggregated Serving |
| **Coder** | Python/CUDA/PyTorch 开发，模型集成、自定义 pipeline、算子开发 |
| **Optimizer** | 性能分析与优化，profiling、kernel 调优、量化、cache 策略 |
| **Reviewer** | 代码审查，检查正确性、性能影响、测试覆盖、项目规范 |

### 内置 Tool

| Tool | 说明 |
|------|------|
| **github** | GitHub CLI 封装（PR 管理、Issue、Review） |
| **vllm** | vllm serve/bench 封装（`vllm serve --omni`） |
| **shell** | 安全的 Shell 命令执行（带超时保护） |

## Pipeline 定义

创建 `pipeline.yaml`：

```yaml
name: "HPC Performance Optimization"

agents:
  - name: architect
    model: glm-5.1
    skills: [hpc-design, vllm-config]
  - name: coder
    model: glm-5.1
    skills: [model-integration, distributed-code]
  - name: optimizer
    model: glm-5.1
    skills: [profiling, kernel-analysis]
  - name: reviewer
    model: glm-5.1
    skills: [code-review]

edges:
  - from: architect
    to: coder
  - from: coder
    to: optimizer
  - from: optimizer
    to: coder
  - from: coder
    to: reviewer

config:
  human_in_the_loop: true
  max_rounds: 10
```

运行：

```bash
vllm_omni_cli run "Optimize Qwen3-Omni inference on 8xA100" --pipeline pipeline.yaml
```

## 编写插件

### 自定义 Tool

```python
# my_tool.py
from vllm_omni_cli.core.tool import BaseTool

class NpuProfileTool(BaseTool):
    name = "npu-profile"
    description = "华为 NPU 性能分析"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要分析的推理命令"},
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs):
        # 实现你的逻辑
        return f"Profile result for: {kwargs['command']}"
```

在 `pyproject.toml` 中注册：

```toml
[project.entry-points."vllm_omni_cli.tools"]
npu-profile = "my_tool:NpuProfileTool"
```

### 自定义 Agent

```python
# my_agent.py
from vllm_omni_cli.core.agent import BaseAgent

class SecurityReviewer(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="security-reviewer",
            role="You are a security expert specializing in ML inference.",
            **kwargs,
        )
```

注册：

```toml
[project.entry-points."vllm_omni_cli.agents"]
security-reviewer = "my_agent:SecurityReviewer"
```

### 自定义 Skill（SKILL.md 格式）

创建目录并编写 `SKILL.md`：

```
my-skill/
└── SKILL.md
```

```markdown
---
name: cuda-tuning
description: CUDA kernel optimization patterns
tools: [shell, vllm]
---

# CUDA Tuning

## Patterns
- Use shared memory for reduction
- Prefer coalesced memory access
- Use warp-level primitives (shfl_sync)
```

加载：

```bash
vllm_omni_cli skill install /path/to/my-skill
```

## 配置

配置文件位于 `~/.vo/config.toml`：

```toml
[llm]
model = "glm-5.1"
api_key = ""
base_url = "https://open.bigmodel.cn/api/paas/v4"

[tools]
github_token = ""

[skills]
paths = []

[defaults]
agents = ["architect", "coder", "reviewer"]
human_in_the_loop = false
```

## 开发

```bash
pip install -e ".[dev]"
ruff check src/
pytest tests/
```

## 链接

- [vllm-omni](https://github.com/vllm-project/vllm-omni) — 多模态推理引擎
- [vllm-omni-skills](https://github.com/hsliuustc0106/vllm-omni-skills) — AI 辅助技能集合（16 个 Skills）
- [vllm-omni 论文](https://arxiv.org/abs/2602.02204)

## License

Apache 2.0
