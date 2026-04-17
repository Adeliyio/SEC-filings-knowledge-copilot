"""Generator agent — synthesizes an answer from retrieved context using Llama 3.

Produces a draft answer with inline citations referencing the source chunks.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from app.agents.retriever import RetrievedContext
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert financial analyst assistant that answers questions about SEC 10-K filings.

RULES:
1. Answer ONLY based on the provided source documents. Do not use prior knowledge.
2. Use inline citations like [Source 1], [Source 2] to reference your sources.
3. If the sources do not contain enough information to fully answer the question, say so explicitly.
4. For financial figures, always cite the exact source.
5. When comparing across companies, organize your answer clearly (e.g., use a table or bullet points).
6. Be precise with numbers — do not round unless the source rounds.
7. Keep answers concise but thorough."""


@dataclass
class GeneratedResponse:
    """Response from the generator agent."""

    answer: str
    query: str
    sources: list[dict] = field(default_factory=list)
    model: str = ""


def _build_prompt(query: str, context: RetrievedContext) -> list[dict]:
    """Build the chat prompt with context and query."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## Source Documents\n\n{context.context_text}\n\n"
                f"---\n\n## Question\n\n{query}\n\n"
                "Provide a thorough answer with inline citations."
            ),
        },
    ]


def generate(query: str, context: RetrievedContext) -> GeneratedResponse:
    """Generate an answer synchronously.

    Args:
        query: The user's question.
        context: Retrieved context from the retriever agent.

    Returns:
        GeneratedResponse with answer and source citations.
    """
    messages = _build_prompt(query, context)

    response = httpx.post(
        f"{settings.ollama_host}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 8192},
        },
        timeout=300.0,
    )
    response.raise_for_status()
    data = response.json()

    answer = data.get("message", {}).get("content", "")

    return GeneratedResponse(
        answer=answer,
        query=query,
        sources=context.source_citations,
        model=data.get("model", settings.ollama_model),
    )


async def generate_stream(query: str, context: RetrievedContext) -> AsyncIterator[str]:
    """Generate an answer with streaming tokens.

    Yields:
        Individual text tokens as they are generated.
    """
    messages = _build_prompt(query, context)

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0.1, "num_ctx": 8192},
            },
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                import json

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = data.get("message", {}).get("content", "")
                if token:
                    yield token

                if data.get("done", False):
                    break
