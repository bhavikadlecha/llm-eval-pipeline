"""
pytest conftest.py — Fixtures and dataset loading.
Dynamically feeds test_cases.json into pytest.mark.parametrize.
Also registers the final report hook to run after all tests complete.
"""
import json
from pathlib import Path

import pytest

from observability.reporter import generate_final_report

DATA_FILE = Path(__file__).parent.parent / "data" / "test_cases.json"


def pytest_configure(config):
    """Register the asyncio mode."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


def load_test_cases():
    """Load and return all test cases from the JSON dataset."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def test_dataset():
    """Session-scoped fixture: loads dataset once for the entire test run."""
    return load_test_cases()


# ── pytest Hook: generate report after all tests ─────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    """
    Hook called after the entire test session finishes.
    Generates the rich console report + HTML report.
    """
    print("\n")
    generate_final_report()
