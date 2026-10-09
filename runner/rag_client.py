import asyncio
import os
from groq import AsyncGroq

# Using AsyncGroq for ultra-fast non-blocking calls
api_key = os.environ.get("GROQ_API_KEY", "").strip()
if not api_key:
    raise ValueError("Missing GROQ_API_KEY! Please add it to your .env file.")

_client = AsyncGroq(api_key=api_key)

async def call_rag_endpoint(
    input_prompt: str,
    retrieved_context: str,
) -> dict:
    """
    Simulates a REAL Agentic RAG system by calling Groq (Llama 3) to act as the Chatbot.
    We instruct it to answer based on the context.
    """
    prompt = (
        f"You are a company customer support bot. Answer the user's question naturally.\n"
        f"You must use this context to answer: {retrieved_context}\n\n"
        f"User Question: {input_prompt}"
    )

    try:
        chat_completion = await _client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="qwen/qwen3.8-27b",  # Fast lightweight model available on your account
            temperature=0.3,
        )
        response_text = chat_completion.choices[0].message.content
    except Exception as exc:
        response_text = f"AI Error: {str(exc)}"

    return {
        "response": response_text,
        "context": retrieved_context,
    }
