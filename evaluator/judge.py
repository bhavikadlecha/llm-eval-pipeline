"""
LLM-as-a-Judge evaluator using the blazing fast Groq API (Llama 3.3).

Improvements over v1:
  - Switched from Google Gemini to Groq for zero quota issues and sub-second evaluations.
  - SQLite cache integration: skip API call if result already exists locally.
  - Supports three evaluation categories: faithfulness, security, relevance.
"""
import json
import os
import re
from typing import Literal

from dotenv import load_dotenv
from groq import Groq
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

from evaluator.cache import get_cached, set_cached
from evaluator.prompts import (
    FAITHFULNESS_JUDGE_PROMPT,
    SECURITY_JUDGE_PROMPT,
    RELEVANCE_JUDGE_PROMPT,
)
from evaluator.schema import EvaluationResult

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client Setup — Groq API (Lightning Fast Llama 3)
# ---------------------------------------------------------------------------
api_key = os.environ.get("GROQ_API_KEY", "").strip()
if not api_key:
    raise ValueError("Missing GROQ_API_KEY! Please add it to your .env file.")
    
_client = Groq(api_key=api_key)

# We use the largest working model on your specific account tier
JUDGE_MODEL = "openai/gpt-oss-120b"

EvalCategory = Literal["faithfulness", "security", "relevance"]

# ---------------------------------------------------------------------------
# Retry Logic — Handle any rare networking blips
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """
    Return True if the exception looks like a transient network or server error.
    """
    if isinstance(exc, ValueError) and "JSON" in str(exc):
        return True # Retry on truncated JSON
        
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("429", "quota", "rate limit", "503", "500", "400")
    )


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_groq(prompt: str) -> str:
    """
    Make a single Groq API call with forced JSON output.
    """
    chat_completion = _client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an impartial AI Judge. You must output only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=JUDGE_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        max_completion_tokens=1024,
    )
    return chat_completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def _build_prompt(
    category: EvalCategory,
    input_prompt: str,
    retrieved_context: str,
    rag_response: str,
    expected_behavior: str,
) -> str:
    """Select the appropriate strict judge prompt and fill in the variables."""
    template_map = {
        "faithfulness": FAITHFULNESS_JUDGE_PROMPT,
        "security":     SECURITY_JUDGE_PROMPT,
        "relevance":    RELEVANCE_JUDGE_PROMPT,
    }
    return template_map[category].format(
        input_prompt=input_prompt,
        retrieved_context=retrieved_context,
        rag_response=rag_response,
        expected_behavior=expected_behavior,
    )


# ---------------------------------------------------------------------------
# Public Evaluation Entry Point
# ---------------------------------------------------------------------------

def evaluate(
    category: EvalCategory,
    input_prompt: str,
    retrieved_context: str,
    rag_response: str,
    expected_behavior: str,
) -> EvaluationResult:
    """
    Evaluate a RAG system response using the Groq Judge.

    Flow:
      1. Check SQLite cache — if hit, return instantly (no API call).
      2. If miss, call Groq with retry logic.
      3. Parse + validate JSON output via Pydantic.
      4. Store result in SQLite cache for future runs.

    Returns a validated EvaluationResult (Pydantic model).
    """
    # ── Step 1: Cache Lookup ─────────────────────────────────────────────────
    cached = get_cached(category, input_prompt, retrieved_context, rag_response)
    if cached is not None:
        logger.info("[CACHE HIT] Skipping Groq call for category=%s", category)
        return cached

    # ── Step 2: Call Groq ────────────────────────────────────────────────────
    prompt = _build_prompt(
        category=category,
        input_prompt=input_prompt,
        retrieved_context=retrieved_context,
        rag_response=rag_response,
        expected_behavior=expected_behavior,
    )

    raw_text = _call_groq(prompt)

    # Strip markdown code fences if the model wraps output in ```json ... ```
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    # ── Step 3: Parse + Validate ─────────────────────────────────────────────
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Judge returned non-JSON output: {raw_text!r}"
        ) from exc

    result = EvaluationResult(**data)   # Pydantic validation — raises on bad schema

    # ── Step 4: Cache the Result ─────────────────────────────────────────────
    set_cached(category, input_prompt, retrieved_context, rag_response, result)

    return result
