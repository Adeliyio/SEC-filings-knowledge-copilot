"""Golden evaluation dataset loader and runner.

Loads curated query triplets from golden_dataset.json and runs them
through the full agent pipeline, collecting retrieval and generation
metrics for regression testing.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation.judge import JudgeScore, judge_response, store_judge_score
from app.evaluation.metrics import (
    EvalResult,
    compute_generation_metrics,
    compute_retrieval_metrics,
    context_relevancy_score,
)

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent.parent.parent.parent / "evaluation" / "golden_dataset.json"


@dataclass
class GoldenEntry:
    """A single golden evaluation entry."""

    id: str
    query: str
    category: str  # single_company, cross_company, multi_step, footnote, risk_factor
    expected_answer: str
    expected_sources: list[dict] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    company: str | None = None
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class GoldenResult:
    """Result of running one golden entry through the pipeline."""

    entry_id: str
    query: str
    category: str
    answer: str = ""
    confidence: float = 0.0
    retrieval_metrics: dict = field(default_factory=dict)
    generation_metrics: dict = field(default_factory=dict)
    judge_scores: dict = field(default_factory=dict)
    overall_score: float = 0.0
    latency_seconds: float = 0.0
    passed: bool = False
    error: str | None = None


@dataclass
class GoldenReport:
    """Aggregate report from running the full golden dataset."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    avg_overall_score: float = 0.0
    avg_confidence: float = 0.0
    avg_latency: float = 0.0
    by_category: dict = field(default_factory=dict)
    results: list[GoldenResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": round(self.passed / max(self.total, 1), 4),
            "avg_overall_score": round(self.avg_overall_score, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_latency_seconds": round(self.avg_latency, 2),
            "by_category": self.by_category,
            "results": [
                {
                    "entry_id": r.entry_id,
                    "query": r.query,
                    "category": r.category,
                    "overall_score": r.overall_score,
                    "confidence": r.confidence,
                    "passed": r.passed,
                    "error": r.error,
                    "latency_seconds": round(r.latency_seconds, 2),
                }
                for r in self.results
            ],
        }


def load_golden_dataset(path: Path | None = None) -> list[GoldenEntry]:
    """Load golden evaluation entries from JSON file.

    Args:
        path: Path to golden_dataset.json. Defaults to project evaluation/ dir.

    Returns:
        List of GoldenEntry objects.
    """
    p = path or GOLDEN_DATASET_PATH
    if not p.exists():
        logger.warning(f"Golden dataset not found at {p}")
        return []

    data = json.loads(p.read_text())
    entries = data.get("entries", data) if isinstance(data, dict) else data

    results = []
    for i, entry in enumerate(entries):
        results.append(GoldenEntry(
            id=entry.get("id", f"golden_{i+1:03d}"),
            query=entry["query"],
            category=entry.get("category", "general"),
            expected_answer=entry.get("expected_answer", ""),
            expected_sources=entry.get("expected_sources", []),
            expected_chunk_ids=entry.get("expected_chunk_ids", []),
            key_facts=entry.get("key_facts", []),
            company=entry.get("company"),
            difficulty=entry.get("difficulty", "medium"),
        ))

    logger.info(f"Loaded {len(results)} golden entries from {p}")
    return results


def run_golden_entry(entry: GoldenEntry, pass_threshold: float = 0.5) -> GoldenResult:
    """Run a single golden entry through the agent pipeline and evaluate.

    Args:
        entry: The golden evaluation entry.
        pass_threshold: Minimum overall score to pass.

    Returns:
        GoldenResult with all metrics.
    """
    from app.agents.graph import run_query

    result = GoldenResult(
        entry_id=entry.id,
        query=entry.query,
        category=entry.category,
    )

    try:
        start = time.time()
        agent_response = run_query(entry.query)
        result.latency_seconds = time.time() - start

        result.answer = agent_response.answer
        result.confidence = agent_response.confidence

        # Retrieval metrics (if we have ground truth chunk IDs)
        retrieved_ids = [s.get("chunk_id", "") for s in agent_response.sources]
        relevant_ids = set(entry.expected_chunk_ids) if entry.expected_chunk_ids else set()

        if relevant_ids:
            ret_metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k=10)
            result.retrieval_metrics = {
                "precision_at_k": ret_metrics.precision_at_k,
                "recall_at_k": ret_metrics.recall_at_k,
                "mrr": ret_metrics.mrr,
            }

        # Generation metrics
        context_text = ""
        for s in agent_response.sources:
            context_text += s.get("text_preview", "") + "\n\n"

        gen_metrics = compute_generation_metrics(
            query=entry.query,
            answer=agent_response.answer,
            context_text=context_text,
            expected_answer=entry.expected_answer if entry.expected_answer else None,
            retrieved_ids=retrieved_ids if relevant_ids else None,
            relevant_ids=relevant_ids if relevant_ids else None,
        )
        result.generation_metrics = {
            "faithfulness": gen_metrics.faithfulness,
            "answer_relevancy": gen_metrics.answer_relevancy,
            "context_precision": gen_metrics.context_precision,
            "context_recall": gen_metrics.context_recall,
        }

        # Judge scoring
        judge = judge_response(entry.query, agent_response.answer, context_text)
        result.judge_scores = judge.to_dict()

        # Store to DB
        store_judge_score(
            query=entry.query,
            response_id=agent_response.message_id,
            session_id=agent_response.session_id,
            judge_score=judge,
            ragas_metrics=result.generation_metrics,
        )

        # Overall score
        scores = [
            gen_metrics.faithfulness,
            gen_metrics.answer_relevancy,
            judge.factual_grounding,
            judge.completeness,
        ]
        result.overall_score = sum(scores) / len(scores) if scores else 0.0
        result.passed = result.overall_score >= pass_threshold

    except Exception as e:
        result.error = str(e)
        logger.error(f"Golden entry {entry.id} failed: {e}")

    return result


def run_golden_dataset(
    entries: list[GoldenEntry] | None = None,
    pass_threshold: float = 0.5,
    categories: list[str] | None = None,
) -> GoldenReport:
    """Run the full golden dataset evaluation suite.

    Args:
        entries: Golden entries to run. Loads from file if None.
        pass_threshold: Minimum score to pass.
        categories: Optional filter by category.

    Returns:
        GoldenReport with aggregate metrics.
    """
    if entries is None:
        entries = load_golden_dataset()

    if categories:
        entries = [e for e in entries if e.category in categories]

    if not entries:
        return GoldenReport()

    report = GoldenReport(total=len(entries))
    total_score = 0.0
    total_confidence = 0.0
    total_latency = 0.0
    category_stats: dict[str, dict] = {}

    for entry in entries:
        logger.info(f"Running golden entry {entry.id}: {entry.query[:60]}...")
        result = run_golden_entry(entry, pass_threshold)
        report.results.append(result)

        if result.error:
            report.errors += 1
        elif result.passed:
            report.passed += 1
        else:
            report.failed += 1

        total_score += result.overall_score
        total_confidence += result.confidence
        total_latency += result.latency_seconds

        # Category tracking
        cat = result.category
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "total_score": 0.0}
        category_stats[cat]["total"] += 1
        if result.passed:
            category_stats[cat]["passed"] += 1
        category_stats[cat]["total_score"] += result.overall_score

    report.avg_overall_score = total_score / max(len(entries), 1)
    report.avg_confidence = total_confidence / max(len(entries), 1)
    report.avg_latency = total_latency / max(len(entries), 1)

    for cat, stats in category_stats.items():
        report.by_category[cat] = {
            "total": stats["total"],
            "passed": stats["passed"],
            "pass_rate": round(stats["passed"] / max(stats["total"], 1), 4),
            "avg_score": round(stats["total_score"] / max(stats["total"], 1), 4),
        }

    logger.info(
        f"Golden eval complete: {report.passed}/{report.total} passed "
        f"(avg_score={report.avg_overall_score:.3f})"
    )

    return report
