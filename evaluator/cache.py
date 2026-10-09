"""
SQLite-based local caching layer for LLM Judge evaluations.

Prevents redundant Gemini API calls on re-runs by storing results
keyed on a SHA-256 hash of (category + prompt + context + rag_response).

Uses Python's built-in sqlite3 — zero extra dependencies, zero cost.
"""
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from evaluator.schema import EvaluationResult

# Store the cache DB alongside this module
CACHE_DB_PATH = Path(__file__).parent / "eval_cache.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS eval_cache (
    cache_key      TEXT PRIMARY KEY,
    passed_test    INTEGER NOT NULL,
    confidence     REAL    NOT NULL,
    reasoning      TEXT    NOT NULL,
    created_at     TEXT    NOT NULL
);
"""


def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite DB and ensure the table exists."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def _make_key(
    category: str,
    input_prompt: str,
    retrieved_context: str,
    rag_response: str,
) -> str:
    """
    Create a deterministic SHA-256 cache key from the evaluation inputs.
    If any single character differs, the key is different.
    """
    raw = json.dumps(
        {
            "category": category,
            "input_prompt": input_prompt,
            "retrieved_context": retrieved_context,
            "rag_response": rag_response,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached(
    category: str,
    input_prompt: str,
    retrieved_context: str,
    rag_response: str,
) -> Optional[EvaluationResult]:
    """
    Look up a cached EvaluationResult.
    Returns None if no cache entry exists (triggers a live API call).
    """
    key = _make_key(category, input_prompt, retrieved_context, rag_response)
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT passed_test, confidence, reasoning FROM eval_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()

    if row is None:
        return None

    return EvaluationResult(
        passed_test=bool(row[0]),
        confidence_score=row[1],
        reasoning_log=row[2],
    )


def set_cached(
    category: str,
    input_prompt: str,
    retrieved_context: str,
    rag_response: str,
    result: EvaluationResult,
) -> None:
    """
    Persist an EvaluationResult to the local SQLite cache.
    Uses INSERT OR REPLACE to handle reruns gracefully.
    """
    key = _make_key(category, input_prompt, retrieved_context, rag_response)
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_cache
                (cache_key, passed_test, confidence, reasoning, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                key,
                int(result.passed_test),
                result.confidence_score,
                result.reasoning_log,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
