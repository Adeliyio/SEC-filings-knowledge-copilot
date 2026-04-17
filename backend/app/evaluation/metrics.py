"""Retrieval and generation evaluation metrics (RAGAs-inspired).

Retrieval metrics:
- Precision@k: fraction of retrieved chunks that are relevant
- Recall@k: fraction of relevant chunks that were retrieved
- MRR (Mean Reciprocal Rank): how high the first relevant chunk ranks
- Context Relevancy: are retrieved chunks useful for answering?

Generation metrics (RAGAs-style):
- Faithfulness: are all generated claims traceable to retrieved chunks?
- Answer Relevancy: does the answer address the user's question?
- Context Precision: are the most relevant chunks ranked highest?
- Context Recall: does the context cover all parts of the ground truth?
"""

import json
import logging
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Retrieval quality scores."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    context_relevancy: float = 0.0
    k: int = 10
    relevant_retrieved: int = 0
    total_retrieved: int = 0
    total_relevant: int = 0


@dataclass
class GenerationMetrics:
    """Generation quality scores (RAGAs-style)."""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


@dataclass
class EvalResult:
    """Combined evaluation result."""

    query: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    overall_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "retrieval": {
                "precision_at_k": self.retrieval.precision_at_k,
                "recall_at_k": self.retrieval.recall_at_k,
                "mrr": self.retrieval.mrr,
                "context_relevancy": self.retrieval.context_relevancy,
                "k": self.retrieval.k,
            },
            "generation": {
                "faithfulness": self.generation.faithfulness,
                "answer_relevancy": self.generation.answer_relevancy,
                "context_precision": self.generation.context_precision,
                "context_recall": self.generation.context_recall,
            },
            "overall_score": self.overall_score,
        }


# --- Retrieval Metrics ---


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 10
) -> float:
    """Fraction of retrieved chunks (top-k) that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Precision@k score (0.0 to 1.0).
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 10
) -> float:
    """Fraction of relevant chunks that were retrieved in top-k.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Recall@k score (0.0 to 1.0).
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: list[str], relevant_ids: set[str]
) -> float:
    """Reciprocal of the rank of the first relevant result.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.

    Returns:
        MRR score (0.0 to 1.0).
    """
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 10
) -> RetrievalMetrics:
    """Compute all retrieval metrics.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        RetrievalMetrics with all scores.
    """
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for cid in top_k if cid in relevant_ids)

    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
        context_relevancy=0.0,  # computed via LLM below
        k=k,
        relevant_retrieved=relevant_retrieved,
        total_retrieved=len(top_k),
        total_relevant=len(relevant_ids),
    )


# --- LLM-Based Metrics ---


def _llm_score(prompt: str, timeout: float = 60.0) -> dict:
    """Call the LLM for a scoring task, expecting JSON output."""
    try:
        response = httpx.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": "You are an evaluation system. Respond with ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 8192},
                "format": "json",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        return json.loads(content)
    except Exception as e:
        logger.warning(f"LLM scoring failed: {e}")
        return {}


def context_relevancy_score(query: str, context_text: str) -> float:
    """Score how relevant the retrieved context is to the query (0-1).

    Uses the LLM to judge whether the context contains information
    needed to answer the query.
    """
    prompt = f"""Rate how relevant the following context is for answering the query.

Query: {query}

Context (truncated):
{context_text[:3000]}

Respond with JSON:
{{"score": <float 0.0 to 1.0>, "reasoning": "<brief explanation>"}}

Scoring guide:
- 1.0: Context contains all information needed to fully answer the query
- 0.7: Context contains most of the needed information
- 0.4: Context is partially relevant but missing key details
- 0.1: Context is barely relevant
- 0.0: Context is completely irrelevant"""

    result = _llm_score(prompt)
    score = result.get("score", 0.0)
    return max(0.0, min(1.0, float(score)))


def faithfulness_score(answer: str, context_text: str) -> float:
    """Score whether all claims in the answer are traceable to the context (0-1).

    Measures hallucination: a score of 1.0 means every claim is grounded.
    """
    prompt = f"""Evaluate the faithfulness of the answer against the source context.
Faithfulness measures whether ALL claims in the answer can be traced back to the context.

Context:
{context_text[:3000]}

Answer:
{answer[:2000]}

Respond with JSON:
{{"score": <float 0.0 to 1.0>, "unsupported_claims": [<list of claims not in context>], "reasoning": "<brief explanation>"}}

Scoring guide:
- 1.0: Every claim in the answer is directly supported by the context
- 0.7: Most claims are supported, minor extrapolations
- 0.4: Several claims are unsupported or contradicted
- 0.0: The answer is mostly fabricated"""

    result = _llm_score(prompt)
    score = result.get("score", 0.0)
    return max(0.0, min(1.0, float(score)))


def answer_relevancy_score(query: str, answer: str) -> float:
    """Score whether the answer actually addresses the query (0-1)."""
    prompt = f"""Evaluate whether the answer directly addresses the question asked.

Question: {query}

Answer:
{answer[:2000]}

Respond with JSON:
{{"score": <float 0.0 to 1.0>, "reasoning": "<brief explanation>"}}

Scoring guide:
- 1.0: Answer fully and directly addresses the question
- 0.7: Answer mostly addresses the question with minor gaps
- 0.4: Answer partially addresses the question, missing key parts
- 0.1: Answer is tangentially related but doesn't address the core question
- 0.0: Answer is completely off-topic"""

    result = _llm_score(prompt)
    score = result.get("score", 0.0)
    return max(0.0, min(1.0, float(score)))


def context_precision_score(
    retrieved_ids: list[str], relevant_ids: set[str]
) -> float:
    """Score whether the most relevant chunks are ranked highest (0-1).

    Computes Average Precision: rewards relevant documents appearing early.
    """
    if not relevant_ids:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            hits += 1
            sum_precision += hits / rank

    if hits == 0:
        return 0.0
    return sum_precision / len(relevant_ids)


def context_recall_score(
    answer: str, context_text: str, expected_answer: str
) -> float:
    """Score whether the context covers all parts of the ground truth answer (0-1).

    Uses LLM to compare expected answer coverage against retrieved context.
    """
    prompt = f"""Evaluate whether the retrieved context contains all the information needed to produce the expected answer.

Expected Answer:
{expected_answer[:2000]}

Retrieved Context (truncated):
{context_text[:3000]}

Respond with JSON:
{{"score": <float 0.0 to 1.0>, "covered_points": [<key facts from expected answer found in context>], "missing_points": [<key facts NOT found in context>], "reasoning": "<brief explanation>"}}

Scoring guide:
- 1.0: Context contains all key facts from the expected answer
- 0.7: Context covers most key facts
- 0.4: Context covers some key facts but misses important ones
- 0.0: Context doesn't contain any of the expected facts"""

    result = _llm_score(prompt)
    score = result.get("score", 0.0)
    return max(0.0, min(1.0, float(score)))


def compute_generation_metrics(
    query: str,
    answer: str,
    context_text: str,
    expected_answer: str | None = None,
    retrieved_ids: list[str] | None = None,
    relevant_ids: set[str] | None = None,
) -> GenerationMetrics:
    """Compute all generation quality metrics.

    Args:
        query: The user query.
        answer: The generated answer.
        context_text: The retrieved context text.
        expected_answer: Ground truth answer (for context_recall).
        retrieved_ids: Ordered retrieved chunk IDs (for context_precision).
        relevant_ids: Ground truth relevant chunk IDs (for context_precision).

    Returns:
        GenerationMetrics with all scores.
    """
    faith = faithfulness_score(answer, context_text)
    relevancy = answer_relevancy_score(query, answer)

    ctx_precision = 0.0
    if retrieved_ids and relevant_ids:
        ctx_precision = context_precision_score(retrieved_ids, relevant_ids)

    ctx_recall = 0.0
    if expected_answer:
        ctx_recall = context_recall_score(answer, context_text, expected_answer)

    return GenerationMetrics(
        faithfulness=faith,
        answer_relevancy=relevancy,
        context_precision=ctx_precision,
        context_recall=ctx_recall,
    )


def compute_full_eval(
    query: str,
    answer: str,
    context_text: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    expected_answer: str | None = None,
    k: int = 10,
) -> EvalResult:
    """Run full evaluation: retrieval + generation metrics.

    Args:
        query: The user query.
        answer: The generated answer.
        context_text: The retrieved context text.
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        expected_answer: Ground truth answer (optional).
        k: Cutoff rank for retrieval metrics.

    Returns:
        EvalResult with all metrics and an overall score.
    """
    # Retrieval metrics
    retrieval = compute_retrieval_metrics(retrieved_ids, relevant_ids, k)
    retrieval.context_relevancy = context_relevancy_score(query, context_text)

    # Generation metrics
    generation = compute_generation_metrics(
        query=query,
        answer=answer,
        context_text=context_text,
        expected_answer=expected_answer,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
    )

    # Overall score: weighted average
    overall = (
        retrieval.precision_at_k * 0.1
        + retrieval.recall_at_k * 0.1
        + retrieval.mrr * 0.1
        + retrieval.context_relevancy * 0.1
        + generation.faithfulness * 0.2
        + generation.answer_relevancy * 0.15
        + generation.context_precision * 0.1
        + generation.context_recall * 0.15
    )

    return EvalResult(
        query=query,
        retrieval=retrieval,
        generation=generation,
        overall_score=round(overall, 4),
    )
