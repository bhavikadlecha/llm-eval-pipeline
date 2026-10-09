"""
Pydantic schema for the LLM-as-a-Judge structured evaluation output.
This schema is what the Gemini Judge is forced to return as structured JSON.
"""
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """
    Structured verdict returned by the Judge LLM for each RAG system response.
    - passed_test: True if the response meets the quality/safety criteria.
    - confidence_score: Judge's certainty (0.0 = not confident, 1.0 = fully confident).
    - reasoning_log: Step-by-step explanation of the verdict.
    """
    passed_test: bool = Field(
        ...,
        description="True if the RAG response passes the evaluation criteria, False otherwise."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The judge's confidence in its verdict, from 0.0 (uncertain) to 1.0 (certain)."
    )
    reasoning_log: str = Field(
        ...,
        description="A concise, step-by-step explanation of why the test passed or failed."
    )
