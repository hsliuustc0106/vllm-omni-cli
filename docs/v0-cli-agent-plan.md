# v0 Plan for a Coherent `vllm-omni-cli` Agent Platform

## Summary

Build a usable `v0` that makes `run`, `run --quick`, and `chat` behave consistently, grounds answers with skills and local model catalogs, and preserves the current CLI surface with small compatibility fixes instead of breaking changes.

This plan explicitly builds on the current codebase and the current `vllm-omni` supported-model surface:

- Keep `LLMFactory` as the model-selection/config entrypoint.
- Keep `Pipeline` as the DAG orchestration primitive for YAML-defined workflows.
- Keep `SkillRegistry` and `SkillAdapter` as the source of truth for installed/loaded skills.
- Add a shared request-preparation layer above these components rather than replacing them.

Model-family scope for `v0` should follow `vllm-omni` itself:

- omni families such as `Qwen3-Omni` and `Qwen2.5-Omni`
- TTS/audio families such as `Qwen3-TTS`
- diffusion and image/video generation families such as `Qwen-Image`, `BAGEL`, `Wan`, `Z-Image`, `HunyuanImage`

Out of scope for `v0`:

- Qwen-VL family routing, prompting, or skill defaults
- generic VLM-chat support that is not part of the current `vllm-omni` supported-model list

Defaults chosen for `v0`:

- Usability over autonomy.
- `run`, `run --quick`, and `chat` are equal-weight workflows.
- Skill selection is deterministic and rules-based, not LLM-decided.
- Catalogs are local-first; remote sync is explicit only.
- CLI compatibility is preserved via aliases/shims, not breaking renames.

## Core Runtime Model

### Shared request object

Introduce a single internal request model named `AgentRequest`. This becomes the normalized input for `run`, `run --quick`, and `chat`.

Location:

- add `src/vllm_omni_cli/core/request.py`
- keep `AgentRequest` and request-preparation helpers there
- do not define request state inside `cli.py`

Required fields:

- `request_id: str`
- `task_text: str`
- `mode: Literal["orchestrated", "quick", "chat"]`
- `target_agents: list[str]`
- `explicit_model: str | None`
- `resolved_model_aliases: list[str]`
- `resolved_model_family: str | None`
- `task_categories: list[str]`
- `hardware_hints: list[str]`
- `manual_skill_refs: list[str]`
- `auto_skill_refs: list[str]`
- `merged_skill_refs: list[str]`
- `tool_scope_agents: list[str]`
- `debug: bool`
- `human_in_the_loop: bool`
- `routing_notes: list[str]`

Inputs vs derived fields:

- direct inputs:
  - `task_text`
  - `mode`
  - `target_agents`
  - `explicit_model`
  - `manual_skill_refs`
  - `debug`
  - `human_in_the_loop`
- derived during preparation:
  - `request_id`
  - `resolved_model_aliases`
  - `resolved_model_family`
  - `task_categories`
  - `hardware_hints`
  - `auto_skill_refs`
  - `merged_skill_refs`
  - `tool_scope_agents`
  - `routing_notes`

Preparation signature:

```python
def prepare_agent_request(
    task_text: str,
    mode: Literal["orchestrated", "quick", "chat"],
    *,
    explicit_model: str | None = None,
    manual_skill_refs: list[str] | None = None,
    target_agents: list[str] | None = None,
    debug: bool = False,
    human_in_the_loop: bool = False,
    skill_registry: SkillRegistry | None = None,
) -> AgentRequest:
    ...
```

Implementation notes:

- `Task` remains the execution-context object, but `AgentRequest` becomes the CLI/runtime normalization object.
- `Task.description` should be populated from `AgentRequest.task_text`.
- `AgentRequest` should be created before any agent or LLM call.
- `request_id` should be generated as a UUID4 string during preparation and included in routing/debug output.
- `tool_scope_agents` is request metadata listing the agent names allowed for the request; actual tool filtering continues to use the existing `ToolRegistry.to_openai_tools(scope=self.name)` behavior inside each agent.
- Add a formatter helper in `core/request.py`, e.g. `render_prepared_task(request: AgentRequest) -> str`, which prepends structured resolver/routing metadata to the original user task.
- For `v0`, do not change `LeadAgent.run(task: str)` or `BaseAgent.run(task: str, ctx: Context)` signatures. The CLI/request layer should render a prepared task string and pass that string into the existing APIs.
- keep `build_model_resolution_note()` as a lower-level utility in `model_catalog.py`
- `render_prepared_task()` becomes the only execution-path renderer used by CLI flows
- `render_prepared_task()` may call `build_model_resolution_note()` internally, but agents should stop calling resolver helpers directly once the shared request pipeline exists
- `ReActAgent.run_single_turn()` and `BaseAgent.chat()` should consume already-rendered prepared task text from the CLI/request layer.
- `Context` and `Task` remain in place. Agents may continue creating `Context(task=Task(description=rendered_task_text))` internally when no context is provided.

### Routing taxonomy

Use a fixed rules-based taxonomy stored in versioned data files, not in the LLM:

- `text-to-image`
- `image-editing`
- `audio-generation`
- `omni-generation`
- `diffusion-serving`
- `serving`
- `hardware-topology`
- `distributed-inference`
- `performance-optimization`
- `code-implementation`
- `code-review`
- `general-architecture`

Classification output may contain multiple categories, but one category should be designated as primary via deterministic precedence.

Routing rules location:

- add `src/vllm_omni_cli/data/routing_rules.json`
- keep taxonomy, precedence, keyword mappings, family-to-skill mappings, and family-to-default-agent mappings in that file
- keep matching logic in Python code

Suggested file shape:

```json
{
  "categories": ["text-to-image", "audio-generation", "omni-generation", "serving"],
  "precedence": ["text-to-image", "audio-generation", "omni-generation", "serving"],
  "keyword_rules": [
    {"keywords": ["latency", "throughput", "ttft", "tokens/s", "perf"], "category": "performance-optimization"},
    {"keywords": ["nvlink", "numa", "pcie", "topology", "l20", "h20", "a100", "h100"], "category": "hardware-topology"}
  ],
  "family_skill_map": {
    "Qwen-Image family": ["vllm-omni-image-gen", "vllm-omni-hardware", "vllm-omni-serving", "vllm-omni-perf", "vllm-omni-recipe"]
  },
  "family_agent_map": {
    "Qwen-Image family": "architect",
    "Qwen-Omni family": "architect",
    "Qwen-TTS family": "architect"
  }
}
```

Matching rules kept in code:

- lowercase all incoming task text before keyword matching
- lowercase all keywords in `routing_rules.json`
- use substring matching for `v0`, not regex
- alias resolution remains exact-match on normalized alias strings
- hardware/model/category keyword matching is case-insensitive

Supported-family scope for routing in `v0`:

- `Qwen-Image family`
- `Qwen-Omni family`
- `Qwen-TTS family`
- additional diffusion/audio/video families imported from the `vllm-omni` supported-model surface

## Implementation Changes

### 1. Add a shared request-preparation layer

Create a single preparation function or module, e.g. `prepare_agent_request(...)`, used by:

- `run`
- `run --quick`
- `chat`

Responsibilities:

- Parse CLI/user input into `AgentRequest`
- Resolve model aliases from builtin and user catalog
- Classify task categories deterministically
- Detect hardware hints from prompt text
- Select default agent(s)
- Select auto skills from installed skills
- Merge manual and auto skill refs
- Generate routing/debug notes
- Render the prepared task text for agent-facing execution

Behavior:

- `run --quick` uses `AgentRequest(mode="quick")`
- `chat` uses `AgentRequest(mode="chat")` every turn, with history appended after preparation
- `run` uses `AgentRequest(mode="orchestrated")`
- default `target_agents` behavior:
  - `quick`: if no explicit agents are provided, use the single primary agent from `family_agent_map`; if no family match exists, fallback to `architect`
  - `chat`: if no explicit `--agent` is provided, fallback to `architect`
  - `orchestrated`: if no explicit agents are provided, keep the current multi-agent default behavior from the CLI path

This layer should not replace `LLMFactory`, `Pipeline`, or `LeadAgent`; it should feed them.

Skill registry lifecycle for `v0`:

- add a helper such as `load_installed_skill_registry() -> SkillRegistry`
- load configured skill roots from `config_get("skills.paths")`
- create and populate the registry in the CLI layer before calling `prepare_agent_request(...)`
- if `skill_registry is None`, `prepare_agent_request(...)` must skip auto-skill selection and append a routing note explaining that no registry was provided
- no global singleton is required in `v0`
- optional in-process caching is allowed later, but should not be required for the first implementation

### 2. Implement deterministic classification and routing

The plan must not rely on vague “intent classification.” For `v0`, implement a deterministic classifier driven by `routing_rules.json`:

- Source of rules:
  - model family alias catalog
  - `routing_rules.json`
- Storage:
  - keep builtin defaults in versioned data files under `src/vllm_omni_cli/data/`
  - allow user catalog overrides only for model aliases, not for routing taxonomy or precedence

Builtin classifier rules shipped in `routing_rules.json` for `v0`:

- `qwen-image`, `qwen-image-edit`, `qwen-image-layered` -> `resolved_model_family = "Qwen-Image family"`
- `qwen3-omni`, `qwen2.5-omni` -> `resolved_model_family = "Qwen-Omni family"`
- `qwen3-tts` -> `resolved_model_family = "Qwen-TTS family"`
- prompts containing `latency`, `throughput`, `ttft`, `tokens/s`, `perf` -> add `performance-optimization`
- prompts containing GPU topology terms like `NVLink`, `NUMA`, `PCIe`, `topology`, `L20`, `H20`, `A100`, `H100` -> add `hardware-topology`
- prompts containing `serve`, `deployment`, `serving`, `vllm`, `api_server` -> add `serving`
- prompts containing `diffusion`, `DiT`, `text-to-image`, `image generation`, `edit image` -> add `text-to-image`
- prompts containing `tts`, `text to speech`, `voice`, `audio generation` -> add `audio-generation`
- prompts containing `omni`, `speech-in speech-out`, `audio-vision-text`, `multimodal generation` -> add `omni-generation`

`hardware_hints` format for `v0`:

- store matched normalized hardware tokens directly, e.g. `["l20", "numa", "pcie"]`
- do not introduce a structured hardware schema in `v0`

Primary-category precedence for `v0`:

1. `text-to-image`
2. `audio-generation`
3. `omni-generation`
4. `serving`
5. `hardware-topology`
6. `performance-optimization`
7. `distributed-inference`
8. `code-implementation`
9. `code-review`
10. `general-architecture`

### 3. Make skill selection deterministic and name-based

Keep path compatibility, but make installed skills a real registry-backed feature:

- Continue `skill install <repo-or-skills-root-path>` storing roots in config.
- Use `SkillRegistry` to index installed skills by:
  - name
  - category
  - source path
  - version
- Add name-based lookup so users can reference installed skills without full filesystem paths.

Skill registry lifecycle for CLI execution:

1. CLI command starts (`run`, `run --quick`, or `chat`)
2. Load configured skill roots from `config_get("skills.paths")`
3. Create a fresh `SkillRegistry`
4. For each configured root path, call `registry.load_from_directory(path)`
5. Pass that populated registry into `prepare_agent_request(...)`

This is the required `v0` bridge between installed skills and deterministic auto-selection.

Skill identity rule for `v0`:

- auto-skill routing matches by the skill `name` field parsed from `SKILL.md`
- do not match by directory name alone
- directory names may still be displayed in UX, but routing must use the parsed skill metadata name
- if a future helper wants friendly aliases, add them explicitly rather than inferring from filesystem layout

This means the routing rules file should store canonical skill metadata names, not directory names.

Auto-skill rules for `v0`:

- `Qwen-Image family`:
  - `vllm-omni-image-gen`
  - `vllm-omni-hardware`
  - `vllm-omni-serving`
  - `vllm-omni-perf`
  - `vllm-omni-recipe`
- `Qwen-Omni family`:
  - `vllm-omni-multimodal`
  - `vllm-omni-distributed`
  - `vllm-omni-serving`
  - `vllm-omni-perf`
  - `vllm-omni-recipe`
- `Qwen-TTS family`:
  - `vllm-omni-audio-tts`
  - `vllm-omni-serving`
  - `vllm-omni-perf`
  - `vllm-omni-hardware`
  - `vllm-omni-recipe`
- Any prompt with hardware topology terms:
  - include `vllm-omni-hardware`

Manual-vs-auto skill merge policy for `v0`:

- auto-selected skills are included first if installed
- manual `--skills` are appended
- de-duplicate by skill name
- if the same skill appears by name and path, prefer the manual reference
- if the same skill appears multiple times in manual input:
  - prefer path-based manual references over name-based manual references
  - otherwise keep the first manual occurrence
- missing auto skills emit warnings and do not fail execution
- missing manual skills fail fast with a clear error

### 4. Define exact `chat` parity for `v0`

`chat` parity should not mean “full orchestration.” For `v0`, define it narrowly:

- `chat` gets:
  - alias resolution
  - deterministic task classification
  - auto-skill selection
  - manual `--skills`
  - `--debug`
  - the same structured resolver block as `quick`
- `chat` does not get:
  - multi-agent orchestration
  - `LeadAgent`
  - DAG `Pipeline`

Implementation behavior:

- keep chat single-agent with history
- before each turn, rebuild an `AgentRequest`
- inject the same routing metadata and skill knowledge as quick mode
- append prior chat history after request preparation
- `chat` should call the same request preparation and task rendering helpers used by `run --quick`

Chat session state for `v0`:

- add a lightweight chat session state object or dict held by the CLI loop
- persist:
  - `resolved_model_family`
  - `resolved_model_aliases`
  - selected skills for the session
  - routing notes useful for later turns
- once a family is resolved in a chat session, keep it sticky across turns unless the user explicitly names a different model family
- if a later message omits the model alias, preparation should reuse the sticky family before falling back to keyword-only classification
- if a later message explicitly names another supported family, replace the sticky family and recompute routing

This prevents:

- turn 1: `deploy qwen-image on 2x L20`
- turn 2: `now benchmark it`

from losing the `Qwen-Image family` context.

`chat()` implementation decision for `v0`:

- do not leave routing solely inside the current `BaseAgent.chat()` implementation
- the CLI `chat` command is responsible for preparing the request and rendering the task before calling `agent.chat(...)`
- update the `chat` command path so it prepares an `AgentRequest`, renders the prepared task string, and passes that rendered string into the agent
- the current `BaseAgent.chat(message, history)` method must be updated or wrapped so it accepts the prepared task text generated by the shared request pipeline
- do not use the raw unprepared user message directly in CLI chat once this plan is implemented

### 5. Standardize grounded metadata format

Replace loose prose-only alias notes with a structured prompt block. For `v0`, use plain text rather than JSON/YAML.

Example format:

```text
[Resolved Model]
alias: qwen-image
family: Qwen-Image family
type: text-to-image / image-editing / DiT
source: vllm-project/recipes

[Routing]
primary_category: text-to-image
extra_categories: hardware-topology, serving, performance-optimization
selected_skills: vllm-omni-image-gen, vllm-omni-hardware, vllm-omni-serving, vllm-omni-perf, vllm-omni-recipe
selected_agent: architect
```

Rules:

- inject this block into `quick` and `chat`
- inject the same block into orchestrated `run` before the `LeadAgent` sees the task
- when `do_not_substitute` is present, the agent must not reinterpret the model family

### 6. Preserve and clarify existing orchestration layers

The plan must explicitly preserve current architecture:

- `LLMFactory`
  - remains the only place that maps agent names/model levels to actual LLM configs
  - request preparation may choose agent names or explicit model, but must not replace factory logic
- `Pipeline`
  - remains the YAML DAG execution path
  - request preparation should populate agents/skills/context before DAG execution, not replace DAG semantics
- `LeadAgent`
  - remains the orchestrator for `run`
  - should receive the prepared task text plus routing metadata block

Exact handoff model for `v0`:

- `prepare_agent_request(...) -> AgentRequest`
- `render_prepared_task(request) -> str`
- `run`:
  - create `AgentRequest`
  - render prepared task string
  - pass rendered string to `LeadAgent.run(...)`
- `run --quick`:
  - create `AgentRequest`
  - render prepared task string
  - pass rendered string to `run_single_turn(...)`
- `chat`:
  - maintain a small session state with sticky resolved family and selected skills
  - for each turn, create `AgentRequest` using current message plus session state
  - render prepared task string
  - pass rendered string to `agent.chat(...)` with history

## Data Flow

The `v0` runtime data flow should be explicit:

```text
CLI command
  -> load config
  -> load SkillRegistry from configured skill roots
  -> prepare_agent_request(...)
  -> render_prepared_task(request)
  -> choose execution path

run
  -> LeadAgent.run(rendered_task)

run --quick
  -> agent.run_single_turn(rendered_task)

chat
  -> maintain chat session state
  -> prepare_agent_request(current_message + sticky session context)
  -> render_prepared_task(request)
  -> agent.chat(rendered_task, history)
```

Context/task wiring:

- `AgentRequest` is not threaded through every downstream method in `v0`
- rendered task text is the handoff boundary into existing agent APIs
- `Task(description=rendered_task)` is the canonical way `AgentRequest` reaches `Task`/`Context`
- `Context` remains unchanged in `v0`

### 7. Clean up command UX with compatibility shims

Preserve current commands and add compatibility improvements:

- keep:
  - `run`
  - `chat`
  - `skill`
  - `catalog`
  - `tool`
  - `config`
  - `list-items`
- add:
  - `list` alias -> forwards to `list-items`
  - `catalog resolve <alias>`
  - `chat --skills`
  - `chat --debug`

Docs/help updates required:

- replace all stale `list skills` examples with valid commands
- show both `list-items` and `list` only if both are intentionally supported
- document that `catalog sync-recipes` is explicit and local-first

## Error Handling and Fallbacks

### Catalog handling

- builtin catalog is always loaded first
- user catalog overrides builtin entries by exact alias match
- if user catalog is missing, continue with builtin catalog
- if user catalog is corrupt/unreadable:
  - log warning
  - ignore user catalog
  - continue with builtin catalog

### Sync handling

- `catalog sync-recipes` failure must not affect runtime `run`, `quick`, or `chat`
- on network/API failure:
  - return non-zero exit for the sync command
  - preserve previous user catalog file if it exists
  - print a short actionable error

### Skill handling

- missing auto-selected skill -> warning only
- missing manual skill -> command error
- empty installed skill registry -> continue with no skills

### Routing fallback

- if no family resolves, classify from keywords only
- if no category matches, use `general-architecture`
- if no auto skills match, continue with zero auto skills

## Packaging and Data Requirements

- commit `src/vllm_omni_cli/data/model_aliases.json`
- ensure package data includes `src/vllm_omni_cli/data/*.json`
- document the role of `vllm-omni-skills` as the primary external skill source for `v0`
- treat `vllm-omni-skills` as an integration target, not a hard runtime dependency

## Test Plan

### Command and UX tests

- `list`, `list-items`, `skill list`, `catalog list` all show consistent output
- README example commands match actual CLI behavior
- `chat` accepts `--skills` and `--debug`
- `catalog resolve qwen-image` shows resolved family and source

### Request preparation tests

- `prepare_agent_request()` produces a complete `AgentRequest`
- `mode` is correct for `run`, `quick`, and `chat`
- hardware hints are extracted from prompts with `L20`, `NUMA`, `PCIe`, `NVLink`

### Routing and resolution tests

- `qwen-image` resolves to `Qwen-Image family`
- `qwen3-omni` resolves to `Qwen-Omni family`
- `qwen3-tts` resolves to `Qwen-TTS family`
- `qwen-image + 2x L20 + latency` produces:
  - primary category `text-to-image`
  - extra categories including `hardware-topology`, `serving`, `performance-optimization`
  - selected agent `architect`
  - image-gen/hardware/serving/perf/recipe skills when installed

### Skill policy tests

- additive merge of auto + manual skills works
- manual duplicate overrides auto-selected path/name conflict
- missing auto skill warns without failing
- missing manual skill fails clearly

### Behavior tests

- `run --quick` and `chat` inject the same structured resolver block
- orchestrated `run` injects the same block before lead-agent execution
- `chat` no longer outputs Qwen-VL or generic VLM assumptions for `qwen-image` prompts
- architect tool scope remains limited after routing unification

### Error-handling tests

- corrupt `~/.vo/model_aliases.json` falls back to builtin catalog
- failed `catalog sync-recipes` preserves prior user catalog
- empty skill installation config still allows run/chat/quick

### Performance sanity tests

- request preparation must stay local and lightweight
- no network call occurs during `run`, `quick`, or `chat`
- sync is isolated to `catalog sync-recipes`

## Assumptions and Defaults

- `vllm-omni-skills` is the primary external skill source for `v0`
- routing and skill selection are deterministic and local
- remote recipe data is opt-in only
- compatibility is preferred over cleanup
- `chat` parity in `v0` means prompt/routing parity, not orchestration parity
- this CLI targets the current `vllm-omni` supported-model surface centered on omni, TTS, and diffusion/image-generation families
