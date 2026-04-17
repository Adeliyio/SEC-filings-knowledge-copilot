"""Feedback loop — connects user feedback to evaluation and retrieval tuning.

- Negative feedback triggers automatic re-evaluation
- Feedback data analyzed for patterns (failure modes, low-scoring queries)
- Retrieval weight tuning based on accumulated feedback signals
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import func, text

from app.config import settings
from app.storage.postgres import (
    EvalScoreRow,
    FeedbackRow,
    get_sync_session_factory,
)

logger = logging.getLogger(__name__)


@dataclass
class FeedbackStats:
    """Aggregate feedback statistics."""

    total_feedback: int = 0
    positive_count: int = 0  # rating >= 4
    negative_count: int = 0  # rating <= 2
    neutral_count: int = 0   # rating == 3
    avg_rating: float = 0.0
    recent_negative: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_feedback": self.total_feedback,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "avg_rating": round(self.avg_rating, 2),
            "positive_rate": round(
                self.positive_count / max(self.total_feedback, 1), 4
            ),
            "negative_rate": round(
                self.negative_count / max(self.total_feedback, 1), 4
            ),
            "recent_negative": self.recent_negative[:10],
        }


@dataclass
class FeedbackTrend:
    """Feedback trends over time."""

    period: str  # "daily", "weekly"
    data_points: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "data_points": self.data_points,
        }


def get_feedback_stats(days: int = 30) -> FeedbackStats:
    """Compute aggregate feedback statistics.

    Args:
        days: Number of days to look back.

    Returns:
        FeedbackStats with counts and averages.
    """
    factory = get_sync_session_factory()
    session = factory()

    try:
        cutoff = datetime.utcnow() - timedelta(days=days)

        rows = (
            session.query(FeedbackRow)
            .filter(FeedbackRow.created_at >= cutoff)
            .all()
        )

        stats = FeedbackStats(total_feedback=len(rows))

        if not rows:
            return stats

        total_rating = 0
        for row in rows:
            total_rating += row.rating
            if row.rating >= 4:
                stats.positive_count += 1
            elif row.rating <= 2:
                stats.negative_count += 1
                stats.recent_negative.append({
                    "id": row.id,
                    "query": row.query,
                    "rating": row.rating,
                    "comment": row.comment or "",
                    "created_at": row.created_at.isoformat() if row.created_at else "",
                })
            else:
                stats.neutral_count += 1

        stats.avg_rating = total_rating / len(rows)
        # Sort negatives by most recent first
        stats.recent_negative.sort(
            key=lambda x: x["created_at"], reverse=True
        )

        return stats

    finally:
        session.close()


def get_feedback_trends(period: str = "daily", days: int = 30) -> FeedbackTrend:
    """Get feedback trends over time.

    Args:
        period: "daily" or "weekly" aggregation.
        days: Number of days to look back.

    Returns:
        FeedbackTrend with time-series data.
    """
    factory = get_sync_session_factory()
    session = factory()

    try:
        cutoff = datetime.utcnow() - timedelta(days=days)

        rows = (
            session.query(FeedbackRow)
            .filter(FeedbackRow.created_at >= cutoff)
            .order_by(FeedbackRow.created_at)
            .all()
        )

        # Group by period
        buckets: dict[str, list] = {}
        for row in rows:
            if not row.created_at:
                continue
            if period == "weekly":
                key = row.created_at.strftime("%Y-W%W")
            else:
                key = row.created_at.strftime("%Y-%m-%d")
            buckets.setdefault(key, []).append(row.rating)

        data_points = []
        for key in sorted(buckets.keys()):
            ratings = buckets[key]
            data_points.append({
                "date": key,
                "count": len(ratings),
                "avg_rating": round(sum(ratings) / len(ratings), 2),
                "positive": sum(1 for r in ratings if r >= 4),
                "negative": sum(1 for r in ratings if r <= 2),
            })

        return FeedbackTrend(period=period, data_points=data_points)

    finally:
        session.close()


def get_eval_score_trends(days: int = 30) -> list[dict]:
    """Get evaluation score trends over time.

    Returns:
        List of daily score aggregates.
    """
    factory = get_sync_session_factory()
    session = factory()

    try:
        cutoff = datetime.utcnow() - timedelta(days=days)

        rows = (
            session.query(EvalScoreRow)
            .filter(EvalScoreRow.created_at >= cutoff)
            .order_by(EvalScoreRow.created_at)
            .all()
        )

        # Group by day
        buckets: dict[str, list[EvalScoreRow]] = {}
        for row in rows:
            if not row.created_at:
                continue
            key = row.created_at.strftime("%Y-%m-%d")
            buckets.setdefault(key, []).append(row)

        trends = []
        for key in sorted(buckets.keys()):
            scores = buckets[key]
            n = len(scores)

            def _safe_avg(vals):
                clean = [v for v in vals if v is not None]
                return round(sum(clean) / len(clean), 4) if clean else None

            trends.append({
                "date": key,
                "count": n,
                "avg_overall": _safe_avg([s.overall_score for s in scores]),
                "avg_faithfulness": _safe_avg([s.faithfulness for s in scores]),
                "avg_answer_relevancy": _safe_avg([s.answer_relevancy for s in scores]),
                "avg_factual_grounding": _safe_avg([s.factual_grounding for s in scores]),
                "avg_completeness": _safe_avg([s.completeness for s in scores]),
                "avg_citation_quality": _safe_avg([s.citation_quality for s in scores]),
                "avg_coherence": _safe_avg([s.coherence for s in scores]),
            })

        return trends

    finally:
        session.close()


def trigger_re_evaluation(response_id: str, query: str) -> dict:
    """Trigger re-evaluation for a response that received negative feedback.

    Re-runs the query through the agent pipeline and scores the new response,
    storing the results for comparison.

    Args:
        response_id: The original response ID that received negative feedback.
        query: The original query.

    Returns:
        Dict with re-evaluation results.
    """
    from app.agents.graph import run_query
    from app.evaluation.judge import judge_and_store

    logger.info(f"Re-evaluating response {response_id}: {query[:60]}...")

    try:
        result = run_query(query)

        judge_score = judge_and_store(
            query=query,
            answer=result.answer,
            context_text="",
            response_id=f"reeval_{response_id}",
            session_id=result.session_id,
        )

        return {
            "status": "completed",
            "original_response_id": response_id,
            "new_response_id": result.message_id,
            "new_confidence": result.confidence,
            "judge_score": judge_score.to_dict(),
        }
    except Exception as e:
        logger.error(f"Re-evaluation failed: {e}")
        return {
            "status": "failed",
            "original_response_id": response_id,
            "error": str(e),
        }


def get_low_scoring_queries(threshold: float = 0.5, limit: int = 20) -> list[dict]:
    """Find queries with consistently low evaluation scores.

    These are candidates for adding to the golden dataset as regression tests.

    Args:
        threshold: Score below which a query is considered low-scoring.
        limit: Maximum number of results.

    Returns:
        List of low-scoring query records.
    """
    factory = get_sync_session_factory()
    session = factory()

    try:
        rows = (
            session.query(EvalScoreRow)
            .filter(EvalScoreRow.overall_score < threshold)
            .order_by(EvalScoreRow.overall_score)
            .limit(limit)
            .all()
        )

        return [
            {
                "query": row.query,
                "response_id": row.response_id,
                "overall_score": row.overall_score,
                "faithfulness": row.faithfulness,
                "answer_relevancy": row.answer_relevancy,
                "factual_grounding": row.factual_grounding,
                "completeness": row.completeness,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            }
            for row in rows
        ]

    finally:
        session.close()
