"""Evaluation API — eval scores, golden dataset, and feedback analytics."""

import logging
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.evaluation.feedback import (
    get_eval_score_trends,
    get_feedback_stats,
    get_feedback_trends,
    get_low_scoring_queries,
    trigger_re_evaluation,
)
from app.evaluation.golden_dataset import (
    load_golden_dataset,
    run_golden_dataset,
    run_golden_entry,
)
from app.evaluation.judge import judge_and_store
from app.storage.postgres import EvalScoreRow, get_sync_session_factory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eval", tags=["evaluation"])


# --- Response Models ---


class EvalScoreResponse(BaseModel):
    id: int
    query: str
    response_id: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    factual_grounding: float | None
    completeness: float | None
    citation_quality: float | None
    coherence: float | None
    overall_score: float | None
    created_at: str | None


class JudgeRequest(BaseModel):
    query: str
    answer: str
    context_text: str
    response_id: str | None = None
    session_id: str | None = None


class JudgeResponse(BaseModel):
    factual_grounding: float
    completeness: float
    citation_quality: float
    coherence: float
    overall: float
    reasoning: dict


class ReEvalRequest(BaseModel):
    response_id: str
    query: str


# --- Endpoints ---


@router.get("/scores", response_model=list[EvalScoreResponse])
def get_eval_scores(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[EvalScoreResponse]:
    """Get recent evaluation scores, ordered by most recent first."""
    factory = get_sync_session_factory()
    session = factory()

    try:
        rows = (
            session.query(EvalScoreRow)
            .order_by(EvalScoreRow.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return [
            EvalScoreResponse(
                id=row.id,
                query=row.query,
                response_id=row.response_id,
                faithfulness=row.faithfulness,
                answer_relevancy=row.answer_relevancy,
                context_precision=row.context_precision,
                context_recall=row.context_recall,
                factual_grounding=row.factual_grounding,
                completeness=row.completeness,
                citation_quality=row.citation_quality,
                coherence=row.coherence,
                overall_score=row.overall_score,
                created_at=row.created_at.isoformat() if row.created_at else None,
            )
            for row in rows
        ]
    finally:
        session.close()


@router.get("/scores/trends")
def get_score_trends(days: int = Query(30, ge=1, le=365)) -> list[dict]:
    """Get evaluation score trends over time (daily aggregates)."""
    return get_eval_score_trends(days=days)


@router.get("/feedback/stats")
def feedback_stats(days: int = Query(30, ge=1, le=365)) -> dict:
    """Get aggregate feedback statistics."""
    stats = get_feedback_stats(days=days)
    return stats.to_dict()


@router.get("/feedback/trends")
def feedback_trends(
    period: str = Query("daily", pattern="^(daily|weekly)$"),
    days: int = Query(30, ge=1, le=365),
) -> dict:
    """Get feedback trends over time."""
    trend = get_feedback_trends(period=period, days=days)
    return trend.to_dict()


@router.post("/judge", response_model=JudgeResponse)
def run_judge(request: JudgeRequest) -> JudgeResponse:
    """Run the LLM judge on a single response and store the result."""
    score = judge_and_store(
        query=request.query,
        answer=request.answer,
        context_text=request.context_text,
        response_id=request.response_id,
        session_id=request.session_id,
    )

    return JudgeResponse(
        factual_grounding=score.factual_grounding,
        completeness=score.completeness,
        citation_quality=score.citation_quality,
        coherence=score.coherence,
        overall=score.overall,
        reasoning=score.reasoning,
    )


@router.post("/re-evaluate")
def re_evaluate(request: ReEvalRequest) -> dict:
    """Trigger re-evaluation for a response (after negative feedback)."""
    return trigger_re_evaluation(
        response_id=request.response_id,
        query=request.query,
    )


@router.get("/golden-dataset")
def get_golden_dataset() -> dict:
    """Get the golden evaluation dataset metadata."""
    entries = load_golden_dataset()
    categories = {}
    for e in entries:
        categories.setdefault(e.category, 0)
        categories[e.category] += 1

    return {
        "total_entries": len(entries),
        "categories": categories,
        "entries": [
            {
                "id": e.id,
                "query": e.query,
                "category": e.category,
                "company": e.company,
                "difficulty": e.difficulty,
            }
            for e in entries
        ],
    }


@router.post("/golden-dataset/run")
def run_golden_eval(
    category: str | None = Query(None),
    limit: int = Query(50, ge=1, le=100),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
) -> dict:
    """Run the golden evaluation dataset (or a subset by category).

    This is a long-running endpoint — each entry goes through the full
    agent pipeline plus evaluation scoring.
    """
    entries = load_golden_dataset()

    if category:
        entries = [e for e in entries if e.category == category]

    entries = entries[:limit]

    if not entries:
        return {"error": "No matching entries found"}

    report = run_golden_dataset(
        entries=entries,
        pass_threshold=threshold,
    )

    return report.to_dict()


@router.get("/low-scoring")
def low_scoring_queries(
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Get queries with consistently low evaluation scores.

    These are candidates for adding to the golden dataset.
    """
    return get_low_scoring_queries(threshold=threshold, limit=limit)
