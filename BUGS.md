# Bug Report — agentic-data-scientist

Observed during the demand-forecasting run (2026-03-01 / 2026-03-02).  
Log source: `agentic_output/demand-forecasting/.agentic_ds.log`

---

## BUG-001 — JSON code-fence wrapping breaks structured output parsing

**Severity:** High  
**Component:** `src/agentic_data_scientist/agents/adk/loop_detection.py` → `_maybe_save_output_to_state`  
**Affected agents:** `success_criteria_checker`, `stage_reflector`  
**Model exhibiting the behaviour:** `google/gemini-2.0-flash-001`

### Description

When an agent has `output_schema` set (Pydantic model), ADK calls
`output_schema.model_validate_json(result)` on the raw LLM text.
`gemini-2.0-flash-001` wraps its JSON response in ` ```json … ``` ` fences
even when the schema is enforced, causing Pydantic to reject the string.

### Log evidence (repeats for every stage)

```
[success_criteria_checker] Error saving output to state: 1 validation error for CriteriaCheckerOutput
  Invalid JSON: expected value at line 1 column 1
  [type=json_invalid, input_value='```json\n{\n  "criteria_...d."\n    }\n  ]\n}\n```', input_type=str]
    For further information visit https://errors.pydantic.dev/2.12/v/json_invalid
  File "loop_detection.py", line 97, in _maybe_save_output_to_state
    self._LlmAgent__maybe_save_output_to_state(event)
  File "llm_agent.py", line 774, in __maybe_save_output_to_state
    result = self.output_schema.model_validate_json(result).model_dump(...)
pydantic_core._pydantic_core.ValidationError: 1 validation error for CriteriaCheckerOutput

[stage_reflector] Error saving output to state: 1 validation error for StageReflectorOutput
  Invalid JSON: expected value at line 1 column 1
  [input_value='```json\n{\n  "stage_mod...new_stages": []\n}\n```', input_type=str]
```

### Impact

- `criteria_checker_output` and `stage_reflector_output` are never written to session state.
- The orchestrator callbacks log "No output found in state" and skip all updates.
- **Criteria remain permanently at 0/5 met**, so the stage orchestrator never exits early.
- All defined stages are executed in full even when the work is already done, wasting API tokens and time.

```
[StageOrchestrator] Criteria status after check: 0/5 met   ← after stage 1
[StageOrchestrator] Criteria status after check: 0/5 met   ← after stage 2
[StageOrchestrator] Criteria status after check: 0/5 met   ← after stage 3
[StageOrchestrator] No remaining stages but criteria not met. Asking reflector to extend stages.
```

### Suggested fix

In `_maybe_save_output_to_state`, strip ` ```json\n` / `\n``` ` fences from the
event text parts **before** delegating to the parent's private method:

```python
def _maybe_save_output_to_state(self, event: Event) -> None:
    if self.output_schema:
        # Strip ```json ... ``` or ``` ... ``` fences added by some models
        for part in (event.content.parts if event.content else []):
            if hasattr(part, 'text') and part.text:
                text = part.text.strip()
                if text.startswith('```json'):
                    part.text = text[7:].rstrip('`').strip()
                elif text.startswith('```'):
                    part.text = text[3:].rstrip('`').strip()
    # ... then call parent
```

---

## BUG-002 — summary_agent hallucinates non-existent `write_file_bound` tool, crashing the run

**Severity:** High  
**Component:** `src/agentic_data_scientist/prompts/base/summary.md` and  
`src/agentic_data_scientist/agents/adk/agent.py` (summary_agent tool list)

### Description

The summary prompt instructs the agent to *"use tools to write the summary markdown
file as `summary.md`"*, but `summary_agent` is registered with only read-only tools:

```
read_file_bound, read_media_file_bound, list_directory_bound,
directory_tree_bound, search_files_bound, get_file_info_bound, fetch_url
```

No write tool exists anywhere in `tools/file_ops.py`. The model invents
`write_file_bound` and calls it, which raises an unhandled `ValueError` that
propagates all the way up through ADK's runner stack and crashes the process.

### Log evidence

```
Error collecting responses: Tool 'write_file_bound' not found.
Available tools: read_file_bound, read_media_file_bound, list_directory_bound,
                 directory_tree_bound, search_files_bound, get_file_info_bound, fetch_url

  File "base_agent.py", line 294, in run_async
  File "loop_detection.py", line 209, in _run_async_impl
  File "functions.py", line 729, in _get_tool
ValueError: Tool 'write_file_bound' not found.
```

Note: `LoopDetectionAgent._run_async_impl` already catches unknown-tool errors and
converts them to a graceful warning event (`_parse_unknown_tool_error`), but this
exception is raised **inside** `self._llm_flow.run_async(ctx)` which is what the
`async for` iterates — the `except` block only catches exceptions that escape the
iterator, not ones raised during `handle_function_calls_async`. The exception
therefore bypasses the handler entirely.

### Impact

- The run crashes at the final stage with exit code 1.
- `summary.md` is never written.
- Bug-003 (below) is triggered as a side effect.

### Suggested fixes

**Option A (preferred):** Add a `write_file` function to `tools/file_ops.py` and
expose it as `write_file_bound` in the agent's tool list (with sandbox path
validation, same as the read tools).

**Option B:** Remove the "write to summary.md" instruction from
`prompts/base/summary.md` so the model only reads files and returns a text
response. The orchestrator already prints agent output to stdout.

---

## BUG-003 — Working directory is deleted even when `--keep-files` is set, if the run crashes

**Severity:** High  
**Component:** `src/agentic_data_scientist/core/api.py` — cleanup logic  

### Description

When `--keep-files` is used, the expectation is that output files survive the run.
However, when BUG-002 causes an unhandled exception, the cleanup code in `api.py`
runs and deletes the working directory regardless of the `keep_files` flag.

### Log evidence

```
Files will be preserved after completion          ← startup, --keep-files respected
...
Cleaned up working directory: .../agentic_output/demand-forecasting   ← crash path ignores flag
```

The second run (08:49 session) also exited with code 1 and the directory was empty
immediately after, confirming the cleanup runs on the exception path.

### Impact

All generated artefacts (trained models, `submission.csv`, engineered CSVs,
workflow scripts) are deleted when any exception occurs, even if `--keep-files`
was requested. The previous successful run's outputs (CV SMAPE 13.00%,
45 000-row submission) were permanently lost this way.

### Suggested fix

In `api.py`, ensure the cleanup block only runs when keep_files is `False`:

```python
finally:
    if not keep_files:
        shutil.rmtree(working_dir, ignore_errors=True)
        logger.info(f"Cleaned up working directory: {working_dir}")
    else:
        logger.info(f"Working directory preserved at: {working_dir}")
```

The `finally` block must not be in the same scope as an exception re-raise that
bypasses the flag check.

---

## Summary table

| ID | Component | Severity | Effect |
|----|-----------|----------|--------|
| BUG-001 | `loop_detection.py` `_maybe_save_output_to_state` | High | Criteria always 0/N met → all stages run to max iterations, wastes tokens |
| BUG-002 | `prompts/base/summary.md` + `agent.py` summary_agent tools | High | Process crash at final stage, no summary written |
| BUG-003 | `core/api.py` cleanup path | High | All outputs deleted on crash even with `--keep-files` |

BUG-001 and BUG-002 compound: BUG-001 causes the orchestrator to believe no
criteria are ever met, BUG-002 then crashes the summary, and BUG-003 ensures
nothing survives. Fixing BUG-001 alone would already substantially reduce wasted
API cost.
