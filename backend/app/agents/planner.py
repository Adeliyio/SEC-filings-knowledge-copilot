"""Planner agent — decomposes user queries into sub-questions and selects retrieval strategy.

Analyzes whether a query targets a single company, multiple companies, or requires
multi-step reasoning, then produces a query plan.
"""

import json
import logging
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

KNOWN_COMPANIES = ["Apple", "Meta", "Microsoft"]

PLANNER_PROMPT = """You are a financial query planner. Given a user question about SEC 10-K filings, produce a JSON query plan.

Available companies: Apple, Meta, Microsoft

Your job:
1. Determine which companies the query targets.
2. Break complex queries into simpler sub-questions if needed.
3. Identify if the query requires cross-company comparison.

Return ONLY valid JSON with this structure:
{
  "strategy": "single_company" | "multi_company" | "comparison",
  "sub_queries": [
    {
      "query": "the sub-question text",
      "company_filter": "Apple" | "Meta" | "Microsoft" | null
    }
  ],
  "synthesis_instruction": "How to combine sub-query results into a final answer (only for comparison/multi_company)"
}

Rules:
- For single-company questions, produce one sub_query with the company_filter set.
- For comparison questions, produce one sub_query per company plus a synthesis instruction.
- If no specific company is mentioned, set company_filter to null.
- Keep sub-queries focused and specific.
- synthesis_instruction should be empty string for single_company strategy."""


@dataclass
class SubQuery:
    query: str
    company_filter: str | None = None


@dataclass
class QueryPlan:
    original_query: str
    strategy: str  # "single_company", "multi_company", "comparison"
    sub_queries: list[SubQuery] = field(default_factory=list)
    synthesis_instruction: str = ""
    attempt: int = 1


def _detect_companies(query: str) -> list[str]:
    """Quick heuristic to detect company mentions in a query."""
    query_lower = query.lower()
    found = []
    for company in KNOWN_COMPANIES:
        if company.lower() in query_lower:
            found.append(company)
    # Check common aliases
    if "aapl" in query_lower and "Apple" not in found:
        found.append("Apple")
    if "msft" in query_lower and "Microsoft" not in found:
        found.append("Microsoft")
    if "meta" in query_lower or "facebook" in query_lower:
        if "Meta" not in found:
            found.append("Meta")
    return found


def plan_query(query: str, attempt: int = 1) -> QueryPlan:
    """Create a query plan by analyzing the user's question.

    Uses LLM for complex queries, fast heuristics for simple ones.

    Args:
        query: The user's original question.
        attempt: Current attempt number (for self-correction retries).

    Returns:
        QueryPlan with strategy and sub-queries.
    """
    detected = _detect_companies(query)

    # Try LLM-based planning
    try:
        response = httpx.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": query},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096},
                "format": "json",
            },
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        plan_data = json.loads(content)

        sub_queries = [
            SubQuery(
                query=sq["query"],
                company_filter=sq.get("company_filter"),
            )
            for sq in plan_data.get("sub_queries", [])
        ]

        if sub_queries:
            return QueryPlan(
                original_query=query,
                strategy=plan_data.get("strategy", "single_company"),
                sub_queries=sub_queries,
                synthesis_instruction=plan_data.get("synthesis_instruction", ""),
                attempt=attempt,
            )
    except Exception as e:
        logger.warning(f"LLM planner failed, falling back to heuristics: {e}")

    # Heuristic fallback
    if len(detected) == 0:
        return QueryPlan(
            original_query=query,
            strategy="single_company",
            sub_queries=[SubQuery(query=query, company_filter=None)],
            attempt=attempt,
        )
    elif len(detected) == 1:
        return QueryPlan(
            original_query=query,
            strategy="single_company",
            sub_queries=[SubQuery(query=query, company_filter=detected[0])],
            attempt=attempt,
        )
    else:
        # Multi-company / comparison
        sub_queries = [
            SubQuery(query=f"{query} (focus on {company})", company_filter=company)
            for company in detected
        ]
        return QueryPlan(
            original_query=query,
            strategy="comparison",
            sub_queries=sub_queries,
            synthesis_instruction=f"Compare findings across {', '.join(detected)} and synthesize a unified answer.",
            attempt=attempt,
        )


def reformulate_query(original_query: str, feedback: str, attempt: int) -> QueryPlan:
    """Reformulate a query plan based on critic feedback.

    Used when the critic scores confidence below threshold and the system retries.

    Args:
        original_query: The original user question.
        feedback: Critic's feedback on why the answer was low-confidence.
        attempt: Current attempt number.

    Returns:
        New QueryPlan with reformulated sub-queries.
    """
    reformulation_prompt = f"""The following question was answered but the answer had low confidence.

Original question: {original_query}

Feedback from quality check: {feedback}

Reformulate the question to get better search results. Be more specific.
Return ONLY valid JSON with the same structure as before."""

    try:
        response = httpx.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": reformulation_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 4096},
                "format": "json",
            },
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        plan_data = json.loads(content)

        sub_queries = [
            SubQuery(
                query=sq["query"],
                company_filter=sq.get("company_filter"),
            )
            for sq in plan_data.get("sub_queries", [])
        ]

        if sub_queries:
            return QueryPlan(
                original_query=original_query,
                strategy=plan_data.get("strategy", "single_company"),
                sub_queries=sub_queries,
                synthesis_instruction=plan_data.get("synthesis_instruction", ""),
                attempt=attempt,
            )
    except Exception as e:
        logger.warning(f"Reformulation failed: {e}")

    # Fallback: just retry the original plan with attempt incremented
    return plan_query(original_query, attempt=attempt)
