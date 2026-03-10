"""
Utility functions and configurations for ADK agents.

This module provides model configuration, helper functions, and shared settings
for the ADK agent system.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.genai import types


load_dotenv()

logger = logging.getLogger(__name__)


# ========================= Instrumented LiteLlm =========================

# Path to JSONL usage log, set by audit harness via env var
_USAGE_LOG_PATH = os.getenv("KDENSE_USAGE_LOG", "")
_usage_call_counter = 0
# Stage context — updated by stage_orchestrator
# Starts as "planning" since planning agents run before orchestrator
_current_stage_name = "planning"
_current_stage_index = -1
_current_agent_name = "unknown"


def set_current_agent(name: str) -> None:
    """Update the current agent name for per-call log attribution."""
    global _current_agent_name
    _current_agent_name = name


def _write_usage_record(record: dict) -> None:
    """Append a usage record to the JSONL log file."""
    if not _USAGE_LOG_PATH:
        return
    try:
        with open(_USAGE_LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.warning(f"[Instrumentation] Failed to write usage record: {e}")


def _write_agent_output(
    call_id: int,
    agent_name: str,
    role: str,
    model: str,
    stage_index: int,
    stage_name: str,
    timestamp: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration: float,
    text: str,
) -> None:
    """Write agent full-text output to a markdown file in agent_outputs/."""
    if not _USAGE_LOG_PATH or not text.strip():
        return
    try:
        out_dir = Path(_USAGE_LOG_PATH).parent / "agent_outputs"
        out_dir.mkdir(exist_ok=True)

        stage_label = f"stage{stage_index:02d}" if stage_index >= 0 else "planning"
        safe_agent = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_name)
        filename = f"call_{call_id:03d}_{stage_label}_{safe_agent}.md"

        content = (
            f"# Call {call_id} — {agent_name}\n\n"
            f"**Stage**: {stage_index} — {stage_name}\n"
            f"**Model**: {model}\n"
            f"**Role**: {role}\n"
            f"**Timestamp**: {timestamp}\n"
            f"**Tokens**: prompt={prompt_tokens} completion={completion_tokens}\n"
            f"**Duration**: {duration:.1f}s\n\n"
            f"---\n\n"
            f"{text}\n"
        )
        (out_dir / filename).write_text(content, encoding="utf-8")
    except Exception as e:
        logger.warning(f"[Instrumentation] Failed to write agent output: {e}")


class InstrumentedLiteLlm(LiteLlm):
    """LiteLlm wrapper that logs per-call token usage and timing to JSONL."""

    _role: str = "unknown"

    def __init__(self, model: str, role: str = "unknown", **kwargs):
        super().__init__(model=model, **kwargs)
        self._role = role

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        global _usage_call_counter
        _usage_call_counter += 1
        call_id = _usage_call_counter

        start_time = time.monotonic()
        start_ts = datetime.now(timezone.utc).isoformat()

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        responses_count = 0
        text_parts: list[str] = []

        try:
            async for response in super().generate_content_async(llm_request, stream=stream):
                responses_count += 1
                if response.usage_metadata:
                    um = response.usage_metadata
                    total_prompt_tokens += getattr(um, 'prompt_token_count', 0) or 0
                    total_completion_tokens += getattr(um, 'candidates_token_count', 0) or 0
                    total_cached_tokens += getattr(um, 'cached_content_token_count', 0) or 0
                # Collect non-thought text parts for agent output log
                if response.content and response.content.parts:
                    for part in response.content.parts:
                        if getattr(part, 'text', None) and not getattr(part, 'thought', False):
                            text_parts.append(part.text)
                yield response
        finally:
            duration = time.monotonic() - start_time

            record = {
                "call_id": call_id,
                "timestamp": start_ts,
                "model": self.model,
                "role": self._role,
                "agent_name": _current_agent_name,
                "stage_index": _current_stage_index,
                "stage_name": _current_stage_name,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "cached_tokens": total_cached_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "duration_seconds": round(duration, 3),
                "stream": stream,
                "responses_count": responses_count,
            }
            _write_usage_record(record)

            # Write full response text to per-call markdown file
            full_text = "\n\n".join(text_parts)
            _write_agent_output(
                call_id=call_id,
                agent_name=_current_agent_name,
                role=self._role,
                model=self.model,
                stage_index=_current_stage_index,
                stage_name=_current_stage_name,
                timestamp=start_ts,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                duration=duration,
                text=full_text,
            )

            logger.info(
                f"[Instrumentation] call={call_id} agent={_current_agent_name} model={self.model} role={self._role} "
                f"prompt={total_prompt_tokens} completion={total_completion_tokens} "
                f"cached={total_cached_tokens} duration={duration:.1f}s"
            )


# Model configuration
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-pro")
REVIEW_MODEL_NAME = os.getenv("REVIEW_MODEL", "google/gemini-2.5-pro")
CODING_MODEL_NAME = os.getenv("CODING_MODEL", "claude-sonnet-4-5-20250929")

logger.info(f"[AgenticDS] DEFAULT_MODEL={DEFAULT_MODEL_NAME}")
logger.info(f"[AgenticDS] REVIEW_MODEL={REVIEW_MODEL_NAME}")
logger.info(f"[AgenticDS] CODING_MODEL={CODING_MODEL_NAME}")

# Configure LiteLLM for OpenRouter
# OpenRouter requires specific environment variables and configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OR_SITE_URL = os.getenv("OR_SITE_URL", "k-dense.ai")
OR_APP_NAME = os.getenv("OR_APP_NAME", "Agentic Data Scientist")

# Export for use in event compression
__all__ = [
    'DEFAULT_MODEL',
    'REVIEW_MODEL',
    'DEFAULT_MODEL_NAME',
    'REVIEW_MODEL_NAME',  # Export model name strings
    'OPENROUTER_API_KEY',
    'OPENROUTER_API_BASE',
    'get_generate_content_config',
    'exit_loop_simple',
    'is_network_disabled',
]

# Set up LiteLLM environment for OpenRouter
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    logger.info("[AgenticDS] OpenRouter API key configured")
else:
    logger.warning("[AgenticDS] OPENROUTER_API_KEY not set - using default credentials")

# Create LiteLLM model instances (instrumented for per-call token/timing logging)
# LiteLLM will automatically route through OpenRouter when model names have the provider prefix (e.g., "google/", "anthropic/")
DEFAULT_MODEL = InstrumentedLiteLlm(
    model=DEFAULT_MODEL_NAME,
    role="orchestration",
    num_retries=10,
    timeout=60,
    # Additional OpenRouter-specific headers
    api_base=OPENROUTER_API_BASE if OPENROUTER_API_KEY else None,
    custom_llm_provider="openrouter" if OPENROUTER_API_KEY else None,
)

REVIEW_MODEL = InstrumentedLiteLlm(
    model=REVIEW_MODEL_NAME,
    role="review",
    num_retries=10,
    timeout=60,
    api_base=OPENROUTER_API_BASE if OPENROUTER_API_KEY else None,
    custom_llm_provider="openrouter" if OPENROUTER_API_KEY else None,
)

# Language requirement (empty for English-only models)
LANGUAGE_REQUIREMENT = ""


def is_network_disabled() -> bool:
    """
    Check if network access is disabled via environment variable.

    Network access is enabled by default. Set DISABLE_NETWORK_ACCESS
    to "true" or "1" to disable network tools.

    Returns
    -------
    bool
        True if network access should be disabled, False otherwise
    """
    disable_network = os.getenv("DISABLE_NETWORK_ACCESS", "").lower()
    return disable_network in ("true", "1")


# DEPRECATED: Use review_confirmation agents instead
# This function is kept for backward compatibility but should not be used in new code.
# Loop exit decisions should be made by dedicated review_confirmation agents with
# structured output and callbacks, not by direct tool calls from review agents.
def exit_loop_simple(tool_context: ToolContext):
    """
    Exit the iterative loop when no further changes are needed.

    DEPRECATED: Use review_confirmation agents instead.

    This function is called by review agents to signal that the iterative
    process should end.

    Parameters
    ----------
    tool_context : ToolContext
        The tool execution context

    Returns
    -------
    dict
        Empty dictionary (tools should return JSON-serializable output)
    """
    tool_context.actions.escalate = True
    return {}


def get_generate_content_config(temperature: float = 0.0, output_tokens: Optional[int] = None):
    """
    Create a GenerateContentConfig with retry settings.

    Parameters
    ----------
    temperature : float, optional
        Sampling temperature (default: 0.0)
    output_tokens : int, optional
        Maximum output tokens

    Returns
    -------
    types.GenerateContentConfig
        Configuration for content generation
    """
    return types.GenerateContentConfig(
        temperature=temperature,
        top_p=0.95,
        seed=42,
        max_output_tokens=output_tokens,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=50,
                initial_delay=1.0,
                max_delay=30,
                exp_base=1.5,
                jitter=0.5,
                http_status_codes=[429, 500, 502, 503, 504],
            )
        ),
    )
