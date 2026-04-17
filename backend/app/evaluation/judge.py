"""LLM-as-Judge scoring pipeline.

Every response is scored by a judge LLM (Llama 3) across four dimensions:
- Factual grounding (0-1): are all claims traceable to retrieved chunks?
- Completeness (0-1): does the answer address the full query?
- Citation quality (0-1): are citations precise and correctly attributed?
- Coherence (0-1): is the answer well-structured and readable?

Scores are logged to PostgreSQL and available via the eval API.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluation judge for a financial question-answering system.
Your job is to score AI-generated answers on four dimensions. Be strict but fair.
You are evaluating answers about SEC 10-K filings from Apple, Meta, and Microsoft.

Respond with ONLY valid JSON."""

JUDGE_PROMPT_TEMPLATE = """Grade an AI answer on 4 dimensions. Use the full 0.0-1.0 range. DO NOT assign the same score to every dimension unless truly warranted.

QUERY: {query}

CONTEXT:
{context}

ANSWER:
{answer}

Scoring anchors (pick granular scores like 0.62, 0.78, 0.91 — avoid only 0.5/0.7/0.9):

factual_grounding — claims in ANSWER are supported by CONTEXT:
  0.9-1.0 all numeric figures and claims match context exactly
  0.7-0.89 most claims supported, 1-2 small gaps or paraphrases
  0.4-0.69 some claims cannot be verified from context
  0.1-0.39 several claims contradict or are absent
  0.0 answer is invented

completeness — ANSWER addresses every part of QUERY:
  0.9-1.0 all sub-questions answered with specifics
  0.7-0.89 main question answered, secondary parts thin
  0.4-0.69 only part of the query addressed
  0.1-0.39 largely off-topic or deflects
  0.0 does not address the query

citation_quality — inline [Source N] citations present and correct:
  0.9-1.0 every numeric claim has a correct [Source N] citation
  0.7-0.89 most claims cited, minor mis-attribution
  0.4-0.69 few citations or wrong sources
  0.1-0.39 citations missing on key claims
  0.0 no citations at all

coherence — answer reads clearly and in logical order:
  0.9-1.0 crisp, well-structured, no repetition
  0.7-0.89 readable with minor flow issues
  0.4-0.69 disorganized or repetitive
  0.1-0.39 hard to follow
  0.0 incoherent

Return ONLY this JSON (one short reason string per dimension, each under 20 words):

{{"factual_grounding": <0.0-1.0>, "fg_reason": "<short reason>", "completeness": <0.0-1.0>, "comp_reason": "<short reason>", "citation_quality": <0.0-1.0>, "cite_reason": "<short reason>", "coherence": <0.0-1.0>, "coh_reason": "<short reason>"}}"""


@dataclass
class JudgeScore:
    """Scores from the LLM judge."""

    factual_grounding: float = 0.0
    completeness: float = 0.0
    citation_quality: float = 0.0
    coherence: float = 0.0
    overall: float = 0.0
    reasoning: dict = None

    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning = {}
        self.overall = round(
            (self.factual_grounding * 0.35
             + self.completeness * 0.25
             + self.citation_quality * 0.20
             + self.coherence * 0.20),
            4,
        )

    def to_dict(self) -> dict:
        return {
            "factual_grounding": self.factual_grounding,
            "completeness": self.completeness,
            "citation_quality": self.citation_quality,
            "coherence": self.coherence,
            "overall": self.overall,
            "reasoning": self.reasoning,
        }


def judge_response(
    query: str,
    answer: str,
    context_text: str,
    timeout: float = 600.0,
) -> JudgeScore:
    """Score a response using the LLM as a judge.

    Args:
        query: The user's original query.
        answer: The generated answer text.
        context_text: The retrieved context the answer was based on.
        timeout: Request timeout in seconds.

    Returns:
        JudgeScore with four dimension scores and an overall score.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        context=context_text[:1500],
        answer=answer[:1500],
    )

    try:
        response = httpx.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                # Raise temperature so a small judge model produces varied,
                # non-collapsing scores. num_predict raised to fit the reasoning
                # strings without truncating the closing brace.
                "options": {"temperature": 0.45, "num_ctx": 3072, "num_predict": 350},
                "format": "json",
            },
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=10.0),
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        result = json.loads(content)

        def _clamp(val):
            return max(0.0, min(1.0, float(val)))

        reasoning = {
            "factual_grounding": result.get("fg_reason", ""),
            "completeness": result.get("comp_reason", ""),
            "citation_quality": result.get("cite_reason", ""),
            "coherence": result.get("coh_reason", ""),
        }

        return JudgeScore(
            factual_grounding=_clamp(result.get("factual_grounding", 0.0)),
            completeness=_clamp(result.get("completeness", 0.0)),
            citation_quality=_clamp(result.get("citation_quality", 0.0)),
            coherence=_clamp(result.get("coherence", 0.0)),
            reasoning=reasoning,
        )
    except Exception as e:
        logger.warning(f"Judge scoring failed: {e}")
        # Surface the failure clearly instead of silently returning 0.0s.
        # `overall` will still compute to 0.0 but reasoning carries the error
        # so the UI/DB row is identifiably a failure rather than a real score.
        return JudgeScore(reasoning={"error": str(e), "status": "judge_failed"})


def store_judge_score(
    query: str,
    response_id: str,
    session_id: str | None,
    judge_score: JudgeScore,
    ragas_metrics: dict | None = None,
) -> None:
    """Persist judge scores + optional RAGAs metrics to PostgreSQL.

    Args:
        query: The user query.
        response_id: The message/response ID.
        session_id: The session ID.
        judge_score: LLM judge scores.
        ragas_metrics: Optional dict with faithfulness, answer_relevancy,
                       context_precision, context_recall.
    """
    from app.storage.postgres import get_sync_session_factory, store_eval_score

    factory = get_sync_session_factory()
    session = factory()

    try:
        score_data = {
            "query": query,
            "response_id": response_id,
            "session_id": session_id,
            # LLM judge scores
            "factual_grounding": judge_score.factual_grounding,
            "completeness": judge_score.completeness,
            "citation_quality": judge_score.citation_quality,
            "coherence": judge_score.coherence,
            # RAGAs metrics (if provided)
            "faithfulness": (ragas_metrics or {}).get("faithfulness"),
            "answer_relevancy": (ragas_metrics or {}).get("answer_relevancy"),
            "context_precision": (ragas_metrics or {}).get("context_precision"),
            "context_recall": (ragas_metrics or {}).get("context_recall"),
            # Overall
            "overall_score": judge_score.overall,
            "details": {
                "reasoning": judge_score.reasoning,
                "ragas": ragas_metrics,
            },
        }
        store_eval_score(session, score_data)
        session.commit()
        logger.info(
            f"Stored judge score for response={response_id}: "
            f"overall={judge_score.overall:.2f}"
        )
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to store judge score: {e}")
        raise
    finally:
        session.close()


def judge_and_store(
    query: str,
    answer: str,
    context_text: str,
    response_id: str | None = None,
    session_id: str | None = None,
    ragas_metrics: dict | None = None,
) -> JudgeScore:
    """Convenience: judge a response and store the result.

    Args:
        query: The user query.
        answer: The generated answer.
        context_text: The retrieved context.
        response_id: Response ID (auto-generated if None).
        session_id: Session ID.
        ragas_metrics: Optional RAGAs-style metrics dict.

    Returns:
        JudgeScore with the evaluation results.
    """
    rid = response_id or str(uuid.uuid4())

    score = judge_response(query, answer, context_text)

    store_judge_score(
        query=query,
        response_id=rid,
        session_id=session_id,
        judge_score=score,
        ragas_metrics=ragas_metrics,
    )

    return score
