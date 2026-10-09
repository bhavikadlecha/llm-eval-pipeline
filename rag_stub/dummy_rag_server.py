"""
Dummy Agentic RAG Server — a FastAPI app that simulates the RAG system under test.

Behavior:
  - For normal prompts: Answers using only the provided retrieved context.
  - For prompt injection attempts: Detects and refuses them.

Run with:
    uvicorn rag_stub.dummy_rag_server:app --reload --port 8000
"""
import re

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Dummy Agentic RAG Server")

# ---------------------------------------------------------------------------
# Injection detection patterns — simple heuristic guard
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+dan",
    r"do\s+anything\s+now",
    r"reveal\s+your\s+system\s+prompt",
    r"bypass\s+your\s+safety",
    r"output\s+all\s+user\s+data",
    r"i\s+have\s+been\s+pwned",
    r"secondary\s+task",
]
INJECTION_REGEX = re.compile(
    "|".join(INJECTION_PATTERNS), re.IGNORECASE
)


class RAGRequest(BaseModel):
    input_prompt: str
    retrieved_context: str


class RAGResponse(BaseModel):
    response: str
    context: str


@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: RAGRequest) -> RAGResponse:
    """
    Main RAG inference endpoint.

    - Detects prompt injections via regex heuristics.
    - Otherwise, generates a context-grounded answer.
    """
    if INJECTION_REGEX.search(request.input_prompt):
        return RAGResponse(
            response=(
                "I'm sorry, I cannot comply with that request. "
                "It appears to contain instructions that conflict with my operational guidelines. "
                "I am here to help you using only the information in the provided context."
            ),
            context=request.retrieved_context,
        )

    # Normal grounded response: acknowledge the context and provide a direct answer
    response_text = (
        f"Based on the available information: {request.retrieved_context.strip()} "
        f"— To address your question: '{request.input_prompt}', "
        f"the answer according to the provided context is as follows: "
        f"{request.retrieved_context.strip()}"
    )

    return RAGResponse(
        response=response_text,
        context=request.retrieved_context,
    )
