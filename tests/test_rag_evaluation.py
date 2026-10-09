"""
Main Test Suite — Automated LLM Evaluation Pipeline (v2).

Improvements over v1:
  - asyncio.Semaphore(2): max 2 concurrent Gemini Judge calls at any time.
    Keeps us well under the free-tier 15 RPM rate limit.
  - Answer Relevance: third evaluation category alongside faithfulness + security.
  - Retries and caching are handled transparently inside evaluator/judge.py.
"""
import asyncio
import json
import pytest
from pathlib import Path

from evaluator.judge import evaluate
from runner.rag_client import call_rag_endpoint
from observability.reporter import log_result

# ---------------------------------------------------------------------------
# Concurrency Limiter
# ---------------------------------------------------------------------------
# Semaphore(5) = at most 5 tests talk to the Groq API at the same time.
# Groq is incredibly fast and has generous limits, so we can run concurrently!
JUDGE_SEMAPHORE = asyncio.Semaphore(5)

# ---------------------------------------------------------------------------
# Load test cases at module level for parametrize
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent.parent / "data" / "test_cases.json"

with open(DATA_FILE, "r", encoding="utf-8") as _f:
    _ALL_CASES = json.load(_f)

_FAITHFULNESS_CASES = [c for c in _ALL_CASES if c["category"] == "faithfulness"]
_SECURITY_CASES     = [c for c in _ALL_CASES if c["category"] == "security"]
_RELEVANCE_CASES    = [c for c in _ALL_CASES if c["category"] == "relevance"]


# ---------------------------------------------------------------------------
# Shared helper: run one evaluation with semaphore protection
# ---------------------------------------------------------------------------

async def _run_evaluation(test_case: dict) -> None:
    """
    Shared logic for all three test categories.
    """
    # Step 1: Call RAG endpoint (local dummy server)
    rag_output = await call_rag_endpoint(
        input_prompt=test_case["input_prompt"],
        retrieved_context=test_case["retrieved_context"],
    )
    rag_response = rag_output["response"]

    # Step 2: Gate entry into the Judge
    async with JUDGE_SEMAPHORE:
        # evaluate() is sync but runs inside an async test
        result = evaluate(
            category=test_case["category"],
            input_prompt=test_case["input_prompt"],
            retrieved_context=test_case["retrieved_context"],
            rag_response=rag_response,
            expected_behavior=test_case["expected_behavior"],
        )

    # Step 3: Log to JSONL + HTML report
    log_result(
        test_id=test_case["id"],
        category=test_case["category"],
        input_prompt=test_case["input_prompt"],
        rag_response=rag_response,
        passed_test=result.passed_test,
        confidence_score=result.confidence_score,
        reasoning_log=result.reasoning_log,
    )

    # Step 4: Assert — let pytest show full reasoning on failure
    assert result.passed_test, (
        f"\n{'='*60}\n"
        f"FAILED [{test_case['category'].upper()}]: {test_case['id']}\n"
        f"  Confidence : {result.confidence_score:.2f}\n"
        f"  Reasoning  : {result.reasoning_log}\n"
        f"  RAG Output : {rag_response[:300]}\n"
        f"{'='*60}"
    )


# ── 1. Faithfulness Tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    _FAITHFULNESS_CASES,
    ids=[c["id"] for c in _FAITHFULNESS_CASES],
)
async def test_faithfulness(test_case: dict):
    """
    Checks that the RAG response is grounded in context only (no hallucination).
    Judge: Gemini 2.0 Flash | Metric: Faithfulness
    """
    await _run_evaluation(test_case)


# ── 2. Security / Prompt Injection Tests ─────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    _SECURITY_CASES,
    ids=[c["id"] for c in _SECURITY_CASES],
)
async def test_security(test_case: dict):
    """
    Checks that the RAG system blocked prompt injection / jailbreak attempts.
    Judge: Gemini 2.0 Flash | Metric: Security / Toxicity
    """
    await _run_evaluation(test_case)


# ── 3. Answer Relevance Tests (NEW) ──────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    _RELEVANCE_CASES,
    ids=[c["id"] for c in _RELEVANCE_CASES],
)
async def test_relevance(test_case: dict):
    """
    Checks that the RAG response actually answers the user's specific question.
    A faithful but off-topic response still fails this test.
    Judge: Gemini 2.0 Flash | Metric: Answer Relevance
    """
    await _run_evaluation(test_case)
