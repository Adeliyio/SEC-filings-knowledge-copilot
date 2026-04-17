"""LangGraph multi-agent orchestration with self-correction loop.

Graph: Planner → Retriever → Generator → Critic
       ↑                                    │
       └──── confidence < 0.65 ─────────────┘
             (reformulate + retry, max 2×)
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, StateGraph

from app.agents.critic import ClaimVerification, CriticResult, critique
from app.agents.generator import GeneratedResponse, generate, generate_stream
from app.agents.planner import QueryPlan, SubQuery, plan_query, reformulate_query
from app.agents.retriever import RetrievedContext, retrieve
from app.config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
CONFIDENCE_THRESHOLD = 0.65

# Background executor for LLM-as-Judge scoring (survives beyond request lifecycle)
_judge_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="judge")


def _background_judge(query: str, answer: str, context_text: str, response_id: str, session_id: str) -> None:
    """Score a response in the background with BOTH LLM-as-judge AND RAGAs metrics.

    Runs:
      - LLM judge: factual_grounding, completeness, citation_quality, coherence
      - RAGAs:     faithfulness, answer_relevancy

    This gives the evaluation row 6 genuinely varied dimensions instead of one
    collapsed score block — which is why earlier rows looked identical. Each
    call uses a different prompt so the scores decorrelate even on Llama 3.2 1B.

    A short sleep lets Ollama settle after the synchronous pipeline finishes,
    which matters on CPU-only hardware where requests queue up serially.
    """
    import time
    try:
        time.sleep(5)
        from app.evaluation.judge import judge_response, store_judge_score
        from app.evaluation.metrics import answer_relevancy_score, faithfulness_score

        # 1) LLM-as-judge (4 dimensions)
        judge_score = judge_response(query=query, answer=answer, context_text=context_text)

        # 2) RAGAs metrics (2 extra independent dimensions). Failures here should
        # not abort storing the judge score.
        ragas: dict = {}
        try:
            ragas["faithfulness"] = faithfulness_score(answer, context_text)
        except Exception as e:
            logger.warning(f"RAGAs faithfulness failed for {response_id}: {e}")
        try:
            ragas["answer_relevancy"] = answer_relevancy_score(query, answer)
        except Exception as e:
            logger.warning(f"RAGAs answer_relevancy failed for {response_id}: {e}")

        store_judge_score(
            query=query,
            response_id=response_id,
            session_id=session_id,
            judge_score=judge_score,
            ragas_metrics=ragas or None,
        )
        logger.info(
            f"Scored response {response_id}: overall={judge_score.overall:.2f} "
            f"faith={ragas.get('faithfulness')} relev={ragas.get('answer_relevancy')}"
        )
    except Exception as e:
        logger.error(f"Background scoring failed for {response_id}: {e}")


# --- Graph State ---


class GraphState(TypedDict, total=False):
    """State passed through the LangGraph pipeline."""

    # Input
    query: str
    session_id: str

    # Planner output
    plan: dict  # serialized QueryPlan
    attempt: int

    # Retriever output
    contexts: list[dict]  # list of serialized RetrievedContext
    context_text: str  # merged context for generator

    # Generator output
    answer: str
    sources: list[dict]

    # Critic output
    confidence: float
    claims: list[dict]  # serialized ClaimVerification list
    grounded_answer: str
    critic_feedback: str

    # Final
    final_answer: str
    final_sources: list[dict]
    final_confidence: float
    final_claims: list[dict]


# --- Node Functions ---


def planner_node(state: GraphState) -> GraphState:
    """Plan the query — decompose into sub-queries with company routing."""
    query = state["query"]
    attempt = state.get("attempt", 1)

    if attempt > 1 and state.get("critic_feedback"):
        logger.info(f"Planner: reformulating query (attempt {attempt})")
        plan = reformulate_query(query, state["critic_feedback"], attempt)
    else:
        logger.info(f"Planner: planning query (attempt {attempt})")
        plan = plan_query(query, attempt)

    logger.info(
        f"Planner: strategy={plan.strategy}, "
        f"sub_queries={len(plan.sub_queries)}"
    )

    return {
        **state,
        "plan": {
            "original_query": plan.original_query,
            "strategy": plan.strategy,
            "sub_queries": [
                {"query": sq.query, "company_filter": sq.company_filter}
                for sq in plan.sub_queries
            ],
            "synthesis_instruction": plan.synthesis_instruction,
            "attempt": plan.attempt,
        },
        "attempt": attempt,
    }


def retriever_node(state: GraphState) -> GraphState:
    """Retrieve context for each sub-query in the plan."""
    from qdrant_client import QdrantClient
    from app.storage.postgres import get_sync_session_factory
    from app.storage.qdrant import get_client as get_qdrant_client

    plan = state["plan"]
    sub_queries = plan.get("sub_queries", [])

    qdrant = get_qdrant_client()
    session_factory = get_sync_session_factory()

    all_contexts = []
    all_chunks_text = []

    for sq in sub_queries:
        db_session = session_factory()
        try:
            ctx = retrieve(
                query=sq["query"],
                qdrant_client=qdrant,
                db_session=db_session,
                top_k=5,
                company_filter=sq.get("company_filter"),
            )
            all_contexts.append({
                "query": ctx.query,
                "company_filter": ctx.company_filter,
                "citations": ctx.source_citations,
                "num_chunks": len(ctx.chunks),
            })
            all_chunks_text.append(ctx.context_text)
        finally:
            db_session.close()

    # Merge context from all sub-queries
    merged_context = "\n\n===\n\n".join(all_chunks_text)

    # Collect all source citations
    all_sources = []
    for ctx_data in all_contexts:
        all_sources.extend(ctx_data.get("citations", []))

    return {
        **state,
        "contexts": all_contexts,
        "context_text": merged_context,
        "sources": all_sources,
    }


def generator_node(state: GraphState) -> GraphState:
    """Generate an answer from the retrieved context."""
    query = state["query"]
    context_text = state.get("context_text", "")
    plan = state.get("plan", {})
    sources = state.get("sources", [])

    # Build a RetrievedContext-like object for the generator
    ctx = RetrievedContext(query=query)

    # If comparison query, add synthesis instruction to the prompt
    synthesis = plan.get("synthesis_instruction", "")
    effective_query = query
    if synthesis:
        effective_query = f"{query}\n\nInstruction: {synthesis}"

    # Generate using direct Ollama call with merged context
    logger.info("Generator: synthesizing answer...")

    system_prompt = """You are an expert financial analyst assistant that answers questions about SEC 10-K filings.

RULES:
1. Answer ONLY based on the provided source documents. Do not use prior knowledge.
2. Use inline citations like [Source 1], [Source 2] to reference your sources.
3. If the sources do not contain enough information to fully answer the question, say so explicitly.
4. For financial figures, always cite the exact source.
5. When comparing across companies, organize your answer clearly (e.g., use a table or bullet points).
6. Be precise with numbers — do not round unless the source rounds.
7. Keep answers concise but thorough."""

    response = httpx.post(
        f"{settings.ollama_host}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"## Source Documents\n\n{context_text}\n\n"
                        f"---\n\n## Question\n\n{effective_query}\n\n"
                        "Provide a thorough answer with inline citations."
                    ),
                },
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
        },
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
    )
    response.raise_for_status()
    answer = response.json().get("message", {}).get("content", "")

    return {
        **state,
        "answer": answer,
    }


def critic_node(state: GraphState) -> GraphState:
    """Evaluate the generated answer for faithfulness and grounding."""
    answer = state.get("answer", "")
    context_text = state.get("context_text", "")
    query = state["query"]

    result = critique(answer, context_text, query)

    return {
        **state,
        "confidence": result.confidence,
        "claims": [
            {
                "claim": v.claim,
                "status": v.status,
                "evidence": v.evidence,
                "reasoning": v.reasoning,
            }
            for v in result.claims
        ],
        "grounded_answer": result.grounded_answer,
        "critic_feedback": result.feedback,
    }


def should_retry(state: GraphState) -> str:
    """Conditional edge: retry if confidence is below threshold and retries remain."""
    confidence = state.get("confidence", 0.0)
    attempt = state.get("attempt", 1)

    if confidence < CONFIDENCE_THRESHOLD and attempt < MAX_RETRIES:
        logger.info(
            f"Confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD}, "
            f"retrying (attempt {attempt + 1}/{MAX_RETRIES})"
        )
        return "retry"
    else:
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD}, "
                f"but max retries ({MAX_RETRIES}) reached"
            )
        return "accept"


def finalize_node(state: GraphState) -> GraphState:
    """Package the final response."""
    return {
        **state,
        "final_answer": state.get("grounded_answer", state.get("answer", "")),
        "final_sources": state.get("sources", []),
        "final_confidence": state.get("confidence", 0.0),
        "final_claims": state.get("claims", []),
    }


def increment_attempt(state: GraphState) -> GraphState:
    """Increment the attempt counter for retry."""
    return {
        **state,
        "attempt": state.get("attempt", 1) + 1,
    }


# --- Build the Graph ---


def build_graph() -> StateGraph:
    """Construct the multi-agent LangGraph pipeline.

    Returns:
        Compiled StateGraph ready to invoke.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("increment_attempt", increment_attempt)

    # Define edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "critic")

    # Conditional: retry or accept
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry": "increment_attempt",
            "accept": "finalize",
        },
    )
    graph.add_edge("increment_attempt", "planner")
    graph.add_edge("finalize", END)

    return graph.compile()


# --- Convenience runners ---


_compiled_graph = None


def get_graph():
    """Get or create the compiled graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


@dataclass
class AgentResponse:
    """Full response from the multi-agent pipeline."""

    answer: str
    confidence: float
    sources: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    attempts: int = 1
    session_id: str = ""
    message_id: str = ""


def run_query(query: str, session_id: str | None = None) -> AgentResponse:
    """Run a query through the full multi-agent pipeline.

    Args:
        query: User's question.
        session_id: Optional session ID for conversation tracking.

    Returns:
        AgentResponse with answer, confidence, sources, and claims.
    """
    sid = session_id or str(uuid.uuid4())
    mid = str(uuid.uuid4())

    # Ensure the session row exists so feedback FK constraints work
    try:
        from app.storage.postgres import ensure_session, get_sync_session_factory
        _sf = get_sync_session_factory()
        _sess = _sf()
        try:
            ensure_session(_sess, sid)
            _sess.commit()
        finally:
            _sess.close()
    except Exception as e:
        logger.warning(f"Failed to create session row: {e}")

    graph = get_graph()

    initial_state: GraphState = {
        "query": query,
        "session_id": sid,
        "attempt": 1,
    }

    logger.info(f"Running query through agent graph: '{query}'")
    result = graph.invoke(initial_state)

    # LLM-as-Judge evaluation (background, non-blocking)
    answer_text = result.get("final_answer", result.get("answer", ""))
    context_text = result.get("context_text", "")
    _judge_executor.submit(
        _background_judge, query, answer_text, context_text, mid, sid
    )

    return AgentResponse(
        answer=result.get("final_answer", result.get("answer", "")),
        confidence=result.get("final_confidence", result.get("confidence", 0.0)),
        sources=result.get("final_sources", result.get("sources", [])),
        claims=result.get("final_claims", result.get("claims", [])),
        plan=result.get("plan", {}),
        attempts=result.get("attempt", 1),
        session_id=sid,
        message_id=mid,
    )


async def run_query_stream(
    query: str, session_id: str | None = None
) -> AsyncIterator[str]:
    """Run a query and stream the response as SSE events.

    Runs planner + retriever synchronously, then streams the generator output,
    followed by critic evaluation.

    Yields:
        SSE-formatted event strings.
    """
    sid = session_id or str(uuid.uuid4())
    mid = str(uuid.uuid4())

    # Ensure the session row exists so feedback FK constraints work
    try:
        from app.storage.postgres import ensure_session, get_sync_session_factory
        _sf = get_sync_session_factory()
        _sess = _sf()
        try:
            ensure_session(_sess, sid)
            _sess.commit()
        finally:
            _sess.close()
    except Exception as e:
        logger.warning(f"Failed to create session row: {e}")

    # Phase 1: Plan
    yield _sse_event("status", {"phase": "planning", "message": "Analyzing query..."})

    plan = await asyncio.to_thread(plan_query, query)

    yield _sse_event("plan", {
        "strategy": plan.strategy,
        "sub_queries": [
            {"query": sq.query, "company_filter": sq.company_filter}
            for sq in plan.sub_queries
        ],
    })

    # Phase 2: Retrieve
    yield _sse_event("status", {"phase": "retrieving", "message": "Searching documents..."})

    from app.storage.postgres import get_sync_session_factory
    from app.storage.qdrant import get_client as get_qdrant_client

    qdrant = get_qdrant_client()
    session_factory = get_sync_session_factory()

    def _retrieve_all(sub_queries):
        """Run retrieval for all sub-queries (sync, runs in thread)."""
        contexts_text = []
        sources = []
        errors = []
        for sq in sub_queries:
            db_session = session_factory()
            try:
                ctx = retrieve(
                    query=sq.query,
                    qdrant_client=qdrant,
                    db_session=db_session,
                    top_k=5,
                    company_filter=sq.company_filter,
                )
                contexts_text.append(ctx.context_text)
                sources.extend(ctx.source_citations)
            except httpx.ConnectError as e:
                errors.append(f"Cannot connect to Ollama at {settings.ollama_host} — is it running? ({e})")
                logger.error(f"Ollama connection failed for sub-query '{sq.query}': {e}")
            except httpx.TimeoutException as e:
                errors.append(f"Ollama request timed out — model may still be loading. ({e})")
                logger.error(f"Timeout during retrieval for sub-query '{sq.query}': {e}")
            except Exception as e:
                error_type = type(e).__name__
                errors.append(f"{error_type}: {e}")
                logger.error(f"Retrieval failed for sub-query '{sq.query}': {e}")
            finally:
                db_session.close()
        return contexts_text, sources, errors

    all_contexts_text, all_sources, retrieval_errors = await asyncio.to_thread(
        _retrieve_all, plan.sub_queries
    )

    merged_context = "\n\n===\n\n".join(all_contexts_text)

    if not all_contexts_text:
        error_detail = "; ".join(retrieval_errors) if retrieval_errors else "No results returned"
        error_msg = f"Retrieval failed: {error_detail}"
        logger.error(error_msg)
        yield _sse_event("status", {"phase": "error", "message": error_msg})
        yield _sse_event("done", {
            "answer": f"Sorry, I was unable to retrieve relevant documents. {error_detail}",
            "confidence": 0.0,
            "sources": [],
            "claims": [],
            "session_id": sid,
            "message_id": mid,
            "attempts": 0,
        })
        return

    yield _sse_event("sources", {"sources": all_sources})

    # Phase 3: Generate (streaming)
    yield _sse_event("status", {"phase": "generating", "message": "Generating answer..."})

    synthesis = plan.synthesis_instruction
    effective_query = query
    if synthesis:
        effective_query = f"{query}\n\nInstruction: {synthesis}"

    # Build a lightweight RetrievedContext for stream generation
    stream_ctx = RetrievedContext(query=effective_query)
    # Override context_text property by injecting into chunks isn't clean,
    # so we stream directly with Ollama

    full_answer = []
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_host}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert financial analyst assistant that answers questions about SEC 10-K filings.\n\n"
                                "RULES:\n"
                                "1. Answer ONLY based on the provided source documents. Do not use prior knowledge.\n"
                                "2. Use inline citations like [Source 1], [Source 2] to reference your sources.\n"
                                "3. If the sources do not contain enough information, say so explicitly.\n"
                                "4. For financial figures, always cite the exact source.\n"
                                "5. When comparing across companies, organize clearly.\n"
                                "6. Be precise with numbers.\n"
                                "7. Keep answers concise but thorough."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"## Source Documents\n\n{merged_context}\n\n"
                                f"---\n\n## Question\n\n{effective_query}\n\n"
                                "Provide a thorough answer with inline citations."
                            ),
                        },
                    ],
                    "stream": True,
                    "options": {"temperature": 0.1, "num_ctx": 4096},
                },
                timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("message", {}).get("content", "")
                    if token:
                        full_answer.append(token)
                        yield _sse_event("token", {"token": token})
                    if data.get("done", False):
                        break
    except httpx.ConnectError as e:
        error_msg = f"Cannot connect to Ollama for generation: {e}"
        logger.error(error_msg)
        yield _sse_event("status", {"phase": "error", "message": error_msg})
        yield _sse_event("done", {
            "answer": f"Sorry, generation failed. {error_msg}",
            "confidence": 0.0, "sources": all_sources,
            "claims": [], "session_id": sid, "message_id": mid, "attempts": 0,
        })
        return
    except httpx.TimeoutException as e:
        error_msg = f"Ollama generation timed out: {e}"
        logger.error(error_msg)
        yield _sse_event("status", {"phase": "error", "message": error_msg})
        yield _sse_event("done", {
            "answer": f"Sorry, generation timed out. {error_msg}",
            "confidence": 0.0, "sources": all_sources,
            "claims": [], "session_id": sid, "message_id": mid, "attempts": 0,
        })
        return
    except Exception as e:
        error_msg = f"Generation failed: {type(e).__name__}: {e}"
        logger.error(error_msg)
        yield _sse_event("status", {"phase": "error", "message": error_msg})
        yield _sse_event("done", {
            "answer": f"Sorry, an error occurred during generation. {error_msg}",
            "confidence": 0.0, "sources": all_sources,
            "claims": [], "session_id": sid, "message_id": mid, "attempts": 0,
        })
        return

    answer_text = "".join(full_answer)

    # Phase 4: Critique
    yield _sse_event("status", {"phase": "verifying", "message": "Verifying claims..."})

    best_answer = answer_text
    best_confidence = 0.0
    best_claims: list[dict] = []
    attempts = 1

    for attempt in range(1, MAX_RETRIES + 1):
        critic_result = await asyncio.to_thread(critique, best_answer, merged_context, query)
        best_confidence = critic_result.confidence
        best_claims = [
            {
                "claim": v.claim,
                "status": v.status,
                "evidence": v.evidence,
                "reasoning": v.reasoning,
            }
            for v in critic_result.claims
        ]

        if best_confidence >= CONFIDENCE_THRESHOLD or attempt >= MAX_RETRIES:
            best_answer = critic_result.grounded_answer
            attempts = attempt
            break

        # Retry: reformulate and regenerate (non-streaming for retries)
        yield _sse_event("status", {
            "phase": "retrying",
            "message": f"Low confidence ({best_confidence:.0%}), reformulating... (attempt {attempt + 1})",
        })

        new_plan = await asyncio.to_thread(reformulate_query, query, critic_result.feedback, attempt + 1)
        # Re-retrieve
        new_contexts_text, new_sources = await asyncio.to_thread(
            _retrieve_all, new_plan.sub_queries
        )

        merged_context = "\n\n===\n\n".join(new_contexts_text)
        all_sources = new_sources

        # Re-generate (non-streaming, in thread to avoid blocking)
        def _regenerate():
            regen_response = httpx.post(
                f"{settings.ollama_host}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert financial analyst assistant that answers questions about SEC 10-K filings.\n\n"
                                "RULES:\n"
                                "1. Answer ONLY based on the provided source documents.\n"
                                "2. Use inline citations like [Source 1], [Source 2].\n"
                                "3. If sources are insufficient, say so.\n"
                                "4. Cite exact sources for financial figures.\n"
                                "5. Organize comparisons clearly.\n"
                                "6. Be precise with numbers.\n"
                                "7. Be concise but thorough."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"## Source Documents\n\n{merged_context}\n\n"
                                f"---\n\n## Question\n\n{query}\n\n"
                                "Provide a thorough answer with inline citations."
                            ),
                        },
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_ctx": 4096},
                },
                timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
            )
            regen_response.raise_for_status()
            return regen_response.json().get("message", {}).get("content", "")

        best_answer = await asyncio.to_thread(_regenerate)

        # Stream the new answer to the client
        yield _sse_event("retry_answer", {"answer": best_answer, "attempt": attempt + 1})

    # Final response
    yield _sse_event("verification", {
        "confidence": best_confidence,
        "claims": best_claims,
        "attempts": attempts,
    })

    # Phase 5: LLM-as-Judge evaluation (runs in background, does not block response)
    _judge_executor.submit(
        _background_judge, query, best_answer, merged_context, mid, sid
    )

    yield _sse_event("done", {
        "answer": best_answer,
        "confidence": best_confidence,
        "sources": all_sources,
        "claims": best_claims,
        "session_id": sid,
        "message_id": mid,
        "attempts": attempts,
    })


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
