# Plan: Agent-Decided Skill Loading

## Summary

This document proposes a change to the current `v0` skill-loading design in `vllm-omni-cli`.

The key shift is:

- routing decides which skills are available
- the agent decides which skills to actually load

This keeps deterministic routing for control and safety, while avoiding prompt bloat from injecting every skill associated with a resolved model family.

## Problem

The current request pipeline and `v0` plan treat family-based auto-skill selection as both:

- an availability filter
- a loading policy

That works for broad discoverability, but it over-injects skill knowledge for narrow tasks.

Example:

- family: `Qwen-Image family`
- mapped skills: `vllm-omni-image-gen`, `vllm-omni-hardware`, `vllm-omni-serving`, `vllm-omni-perf`, `vllm-omni-recipe`
- task: "validate TP=2 latency math on 2xL20"

For this task, only `vllm-omni-perf` is likely needed. Loading all 5 skills adds irrelevant tokens and makes the prompt harder for the agent to use effectively.

The root issue is that skill relevance depends on the task, not only on the family.

## Design Principle

Use a two-stage decision model:

1. Deterministic routing narrows the candidate skill set.
2. The agent dynamically chooses which candidate skills to load.

This changes the role of static routing:

- family and category rules still determine which skills may be offered
- those rules no longer force full skill knowledge into the initial system prompt

## Proposed Architecture

### 1. Keep deterministic routing

Preserve the current deterministic logic for:

- model family resolution
- task category classification
- hardware hint detection
- allowed skill-set derivation

This means `family_skill_map` still has value, but only as an allowlist or candidate selector.

### 2. Replace eager skill injection with skill advertisement

Prepared requests should include compact metadata for each available skill instead of full `SKILL.md` knowledge.

Suggested advertised fields:

- `name`
- `description`
- optional `category`
- optional tags such as `perf`, `serving`, `hardware`
- optional token-size hint

These summaries should be small enough to keep the prompt readable while still telling the agent what capabilities exist.

### 3. Add on-demand skill loading

Expose a `load_skill` tool, or an equivalent internal retrieval path, that lets the agent request the full content of a skill only when needed.

Conceptually:

```text
load_skill("vllm-omni-perf") -> full knowledge for that skill
```

This allows the agent to pull only the relevant knowledge after reading the task.

### 4. Update request preparation semantics

The current `AgentRequest` uses:

- `manual_skill_refs`
- `auto_skill_refs`
- `merged_skill_refs`

Under this proposal, request preparation should distinguish between:

- skills that are available to the agent
- skills that were actually loaded by the agent

Suggested direction:

- keep `manual_skill_refs` for explicit user intent
- rename `auto_skill_refs` semantics to available auto-selected skills
- replace or de-emphasize `merged_skill_refs` as the default execution payload
- add a field such as `available_skill_refs`
- optionally add `advertised_skills` for prompt-rendering metadata

The key rule is that available skills are not automatically equivalent to loaded skills.

### 5. Update prompt rendering

`render_prepared_task()` should render:

- resolved model metadata
- routing metadata
- a compact list of available skills
- short guidance telling the agent to load only the skills it needs

It should not render full skill bodies by default.

### 6. Update agent prompt construction

Today `BaseAgent._build_system_prompt()` appends full `skill.knowledge` for every loaded skill.

That behavior should evolve toward:

- start with role text
- include skill summaries or references when available
- append full knowledge only for skills that have actually been loaded on demand

This keeps the current prompt model intact, but changes when full knowledge enters it.

## Recommended Option

The preferred design is:

- summaries in the initial prompt
- full skill content available on demand
- deterministic routing used as a filter, not as a loader

This preserves the strongest parts of the current architecture:

- predictable routing
- no LLM-based router
- local-first skills

while fixing the over-injection problem.

## Request Flow

Recommended flow for a task such as "validate TP=2 latency math for Qwen-Image on 2xL20":

1. `prepare_agent_request(...)` resolves the family and task categories.
2. Routing rules make several skills available, for example the 5 `Qwen-Image` candidates.
3. `render_prepared_task()` includes only short summaries for those skills.
4. The agent reads the task and decides it only needs `vllm-omni-perf`.
5. The agent calls `load_skill("vllm-omni-perf")`.
6. The full perf knowledge is loaded into context.
7. The agent answers without loading unrelated image-gen, recipe, or serving knowledge.

## Preload Exceptions

Default behavior should be on-demand loading, but eager loading is still reasonable in a few cases:

- the user explicitly names a skill
- the task clearly and narrowly requires one obvious skill
- the environment makes tool round trips significantly more expensive than prompt growth

These should be documented as exceptions, not the default policy.

## Data Model Direction

Suggested `AgentRequest` evolution:

- keep:
  - `manual_skill_refs`
- change meaning of:
  - `auto_skill_refs` from "auto-loaded" to "auto-available"
- add:
  - `available_skill_refs`
  - `advertised_skills`
  - optional `preloaded_skill_refs`
- do not treat:
  - `merged_skill_refs`
  as the default list of full skill payloads injected into the agent prompt

This can be implemented incrementally. The first step does not require a perfect final schema.

## Tooling Direction

Suggested `load_skill` behavior:

- input: canonical skill name
- validation: must be present in the allowed or available skill set for the request, unless explicitly user-provided
- output: full skill knowledge and possibly structured metadata

Possible follow-up features:

- memoize loaded skills within the request
- expose a `list_available_skills` helper for debugging
- track loaded-skill token cost

## Observability

To evaluate whether this design is working, track:

- skills offered to the agent
- skills actually loaded
- unused offered skills
- prompt token savings
- extra latency from skill loads
- cases where the agent failed to load a needed skill

This will show whether the design improves relevance without adding too much round-trip cost.

## Validation Tasks

Use representative tasks to confirm the policy:

- "Deploy Qwen-Image on 2xL20"
  - likely needs `serving`, `hardware`, `perf`
- "Benchmark Qwen-Image latency"
  - likely needs `perf`
- "Write a recipe for Qwen-Image"
  - likely needs `recipe`
- "Validate TP=2 math for Qwen-Image"
  - likely needs `perf`

The success criterion is that the system offers a broad enough candidate set but loads only the subset needed for the task.

## Impact on the Existing v0 Plan

This proposal revises one assumption in `docs/v0-cli-agent-plan.md`:

- current assumption: skill selection is deterministic and rules-based
- proposed refinement: skill availability remains deterministic and rules-based, but skill loading becomes agent-decided

So the updated principle should become:

> Deterministic routing narrows the candidate skill set; dynamic agent judgment determines which skill knowledge is loaded into context.

This change does not require replacing the current routing layer. It changes what that layer returns and how the agent consumes it.

## Migration Strategy

Recommended implementation order:

1. Add this design note and align on the architecture.
2. Introduce advertised-skill metadata in request preparation.
3. Change prompt rendering to show summaries instead of full skill bodies.
4. Add a `load_skill` path for agent-driven retrieval.
5. Update agent execution to append full knowledge only after explicit loading.
6. Add metrics for offered vs loaded skills.
7. Revise the main `v0` plan once the direction is accepted.

This keeps the change incremental and avoids forcing a full rewrite of the current request pipeline.
