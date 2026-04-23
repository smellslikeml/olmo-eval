# Implementing SciCode in olmo-eval

## Goal

Reproduce Artificial Analysis's reported **`sub_step_accuracy ≈ 0.41`** for
`google/gemma-4-31b-it` on the SciCode benchmark. Current implementation on
`finbarr/scicode` scored `0.333` on a `limit=1` Beaker run, but the
investigation below shows that number is spurious — the cascade of sub-step
generations never actually reached vLLM. We need an end-to-end run where the
cascade fires.

## What SciCode is

SciCode is a scientific-coding benchmark with 80 problems (15 validation,
65 test). Each problem has N sub-steps (288 scorable sub-steps total on the
test split). For each sub-step:

1. The model receives a prompt containing: the problem statement, the
   function header for this sub-step, and the **code it previously generated
   for all prior sub-steps of the same problem**.
2. The model emits a Python function.
3. The function is executed against reference test cases with targets loaded
   from an HDF5 file (`test_data.h5`) via pytest-style `assert` lines.

Grading is pure execution (no LLM judge). The cascade across sub-steps is
**same-model, programmatic prompt-chaining, no tools** — prompt `i+1` is a
pure function of the outputs of sub-steps `1..i` and the sub-step metadata.

## Why SciCode is hard to implement in this codebase

olmo-eval cleanly handles three eval shapes. SciCode matches none of them.

| Shape | Examples | Inference pattern | Scoring pattern |
|---|---|---|---|
| **Single-shot + programmatic grading** | MMLU, most code tasks | 1 `agenerate` call per instance | regex / exact-match / pytest |
| **LLM-as-judge** | tau2bench | 1 call on main provider | auxiliary provider (often stronger model) grades in `score_responses` |
| **Agentic / multi-turn** | tool-use tasks | N calls via `harness.run()` loop; model emits tool calls, runtime feeds results back | programmatic, after the loop |
| **SciCode** (odd one out) | — | N sequential same-model calls per instance, chained **by task code** with no tools | programmatic (pytest against HDF5 targets) |

The library has no first-class pattern for "N programmatic same-model
generations driven by task code." The two library-native ways to get N calls
per instance both misfit:

- **Backend loops** (`openai_agents`, `openhands`): require tool-driven turns.
  SciCode's chaining is not tool-driven — the task code structures the next
  prompt from sub-step metadata, not from a model-emitted tool call.
- **Auxiliary providers**: designed for *different*-model judges, not same-
  model continuations. Works, but spawns a second vLLM instance (extra GPUs).

## How the harness setup works

`AsyncEvalRunner.run_async` (`src/olmo_eval/runners/asynq/runner.py:142`)
orchestrates the run across three kinds of processes, communicating over
`mp.Queue`s:

```
                 ┌───────────────────────────┐
                 │    AsyncEvalRunner        │
                 │    (main process)         │
                 └────────────┬──────────────┘
                              │ spawn
         ┌────────────────────┼──────────────────────────┐
         │                    │                          │
         ▼                    ▼                          ▼
┌─────────────────┐  ┌──────────────────┐   ┌────────────────────────┐
│ Inference       │  │ Scoring worker   │   │ InferenceManager       │
│ worker(s)       │  │ (1 process)      │   │ (auxiliary providers)  │
│ (N processes,   │  │                  │   │ — lives in main proc,  │
│  1 per vLLM     │  │ runs             │   │   spawns subprocesses  │
│  instance)      │  │ task.score_      │   │   for each aux vLLM    │
│                 │  │ responses()      │   │                        │
│ owns            │  │                  │   │ providers reachable    │
│ harness.provider│  │ has access to    │   │ via                    │
│ (main vLLM)     │  │ ScoringContext.  │   │ ScoringContext.        │
│                 │  │ inference_pool   │   │ inference_pool         │
└────────┬────────┘  └──────────────────┘   └────────────────────────┘
         │                    ▲
         │   item_queue       │
         ▼                    │
   (instances)                │
         │                    │
         └───► result_queue ──┴───► scoring_queue ──► scored_queue
```

### Key lifecycle facts

1. **Inference workers** (`workers.py`) run the main provider (typically
   `vllm_server`). Each worker's `finally` block calls
   `harness.provider.close()` — which **stops the local vLLM server** — as
   soon as it has drained its share of the item queue. In the current
   Beaker run, this happens ~1 second *before* the scoring worker starts
   processing results.

2. **The scoring worker** (`workers.py:scoring_worker`) runs
   `task.score_responses(...)` in its own subprocess. Crucially, it has
   **no access to the main provider** — the inference worker owned it and
   already tore it down. The scoring worker only sees providers exposed via
   its `registry_config` argument, which is built from
   `harness_config.auxiliary_providers`.

3. **Auxiliary providers** (`runner.py:219-230`) are managed by
   `InferenceManager` in the *main* process. They live for the full run
   (not tied to inference-worker lifetime) and are exposed to the scoring
   worker through `ScoringContext.inference_pool`.

4. **The scorer's view of LLMs** is therefore *exactly* the auxiliary
   providers. The SciCode task's `score_responses` looks up a provider
   named `"cascade"` in `context.inference_pool` — it will never find the
   main provider there.

### Why the current 0.333 result is wrong

`logs/vllm_server.log` shows **1** `POST /v1/chat/completions` request
(problem 44 has 3 sub-steps, 0 hardcoded — should be 3). The experiment log
shows:

- `01:59:16` — inference worker's `finally` fires → `Stopping vLLM server`.
- `01:59:17` — scoring worker starts `score_responses` → tries to call the
  cascade → server is dead.
- Next 25 seconds — `Retrying request to /chat/completions` from an HTTP
  client pointed at a shut-down server.
- Cascade effectively produced no usable continuations; the one sub-step
  that was graded came from the single inference-phase call.

The `0.333` is one problem × one sub-step passing, not the cascade working.

## Constraints for the fix

- **No library changes.** The fix must use existing olmo-eval surface
  (tasks, config, launch flags). No edits to `workers.py`, `runner.py`,
  `processing.py`, `base.py`, etc.
- SciCode task code already looks up a provider named `"cascade"` in
  `ScoringContext.inference_pool` (see `scicode.py:_resolve_provider_name`),
  and raises a clear error message pointing at the `-o` launch flags if the
  provider isn't declared. This was done in commit `2272788`.

## Plan — declare `cascade` as an auxiliary provider at launch

Because auxiliary providers are the only library-native surface exposed to
the scoring worker, we must declare the cascade provider there. Two
sub-options differ only in GPU topology.

### Sub-option A — separate vLLM for the cascade (recommended for first run)

Launch the auxiliary as a second `vllm_server` instance with the same
model. `GPUPlanner` allocates disjoint GPUs for auxiliary providers, so
Gemma 4 31B at TP=2 goes from 2 GPUs (main only) → 4 GPUs (main + aux).

Beaker launch flags to add to the existing command:

```
-o auxiliary_providers.cascade.kind=vllm_server
-o auxiliary_providers.cascade.model=google/gemma-4-31b-it
-o auxiliary_providers.cascade.num_instances=1
-o auxiliary_providers.cascade.kwargs.tensor_parallel_size=2
-o auxiliary_providers.cascade.kwargs.chat_template_kwargs={"enable_thinking":true}
```

Pros: zero orchestration changes; works with the existing gantry flow.
Cons: +2 GPUs (4 total for Gemma 4 31B); +~170s for the second server to
initialize.

### Sub-option B — shared vLLM via external `base_url` (GPU-cost optimization)

`vllm_server` with `base_url` set skips local spawn and makes `close()` a
no-op (see `src/olmo_eval/inference/providers/vllm_server.py:216-251`). So
we can launch one vLLM manually in the Beaker entrypoint, then configure
**both** the main provider and the cascade auxiliary to point at
`http://localhost:8000/v1`. Single server, two logical providers, 2 GPUs
total. Requires wrapping the entrypoint to start vLLM, wait for readiness,
then run `olmo-eval`. This is orchestration, not library change.

Deferred until A proves the cascade fires end-to-end.

## Verification

1. **Beaker `limit=1` validation** with Sub-option A flags.
   - Both vLLM instances log `Provider ready`.
   - Main `logs/vllm_server.log`: 1 chat completion.
   - Auxiliary `logs/vllm_server_*.log`: **2** chat completions for
     problem 44 (3 sub-steps, 1 inference-phase call → 2 cascade calls).
   - No `Retrying request to /chat/completions` errors.
   - `sub_step_accuracy > 0.333`.

2. **Beaker full test split** (65 problems, 288 sub-steps) once (1) is
   healthy. Target: `sub_step_accuracy ≈ 0.35–0.45`.

3. Check `metrics/vllm_server_*.jsonl` for the auxiliary to confirm it
   actually served requests (not idle).

## Files touched / not touched

- **No source edits.** The relevant task code already exists at
  `src/olmo_eval/evals/tasks/scicode.py` (commit `2272788`).
- **No test changes.** 16 unit tests in `tests/evals/tasks/test_scicode.py`
  pass locally; they exercise the cascade against a fake provider pool
  whose only member is named `"cascade"`.
- **Launch-time flags only** — supplied via `-o auxiliary_providers.cascade.*`
  on the Beaker command.
