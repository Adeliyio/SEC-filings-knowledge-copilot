"""Critic agent — verifies answers against retrieved context.

Decomposes answers into atomic claims, checks each claim against source chunks,
and produces a confidence score with per-claim grounding status.
"""

import json
import logging
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

CLAIM_DECOMPOSITION_PROMPT = """You are a claim extraction system. Given an answer about SEC 10-K filings, extract every factual claim as a separate atomic statement.

Rules:
1. Each claim should be a single, verifiable statement.
2. Preserve exact numbers, percentages, and dollar amounts.
3. Include which company each claim is about.
4. Ignore meta-statements like "according to the filing" — focus on the factual content.

Return ONLY valid JSON:
{
  "claims": [
    "Apple's total revenue in FY 2024 was $391.0 billion",
    "Revenue increased 2% compared to FY 2023"
  ]
}"""

CLAIM_VERIFICATION_PROMPT = """You are a fact-checking system. Given a claim and source documents, determine if the claim is supported.

Respond with ONLY valid JSON:
{
  "status": "supported" | "partially_supported" | "unsupported",
  "evidence": "Brief quote or reference from the source that supports or contradicts the claim",
  "reasoning": "Why you gave this status"
}

Rules:
- "supported": The claim is directly stated or clearly derivable from the sources.
- "partially_supported": The claim is roughly correct but has minor inaccuracies (e.g., rounded numbers).
- "unsupported": The claim cannot be verified from the given sources, or contradicts them."""


@dataclass
class ClaimVerification:
    claim: str
    status: str  # "supported", "partially_supported", "unsupported"
    evidence: str = ""
    reasoning: str = ""


@dataclass
class CriticResult:
    """Full critic evaluation of a generated answer."""

    confidence: float  # 0.0 to 1.0
    claims: list[ClaimVerification] = field(default_factory=list)
    grounded_answer: str = ""  # Answer with unsupported claims removed/flagged
    feedback: str = ""  # Summary of issues found
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0


def _llm_json_call(messages: list[dict]) -> dict:
    """Make an LLM call expecting JSON output."""
    response = httpx.post(
        f"{settings.ollama_host}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 4096},
            "format": "json",
        },
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return json.loads(content)


def decompose_claims(answer: str) -> list[str]:
    """Break an answer into atomic factual claims.

    Args:
        answer: The generated answer text.

    Returns:
        List of claim strings.
    """
    try:
        result = _llm_json_call([
            {"role": "system", "content": CLAIM_DECOMPOSITION_PROMPT},
            {"role": "user", "content": f"Answer:\n{answer}"},
        ])
        claims = result.get("claims", [])
        if isinstance(claims, list) and claims:
            return [str(c) for c in claims]
    except Exception as e:
        logger.warning(f"Claim decomposition failed: {e}")

    # Fallback: split by sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def verify_claim(claim: str, context_text: str) -> ClaimVerification:
    """Verify a single claim against the retrieved context.

    Args:
        claim: An atomic factual claim.
        context_text: The formatted context from the retriever.

    Returns:
        ClaimVerification with status and evidence.
    """
    try:
        result = _llm_json_call([
            {"role": "system", "content": CLAIM_VERIFICATION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Source Documents\n\n{context_text}\n\n"
                    f"---\n\n## Claim to Verify\n\n{claim}"
                ),
            },
        ])
        return ClaimVerification(
            claim=claim,
            status=result.get("status", "unsupported"),
            evidence=result.get("evidence", ""),
            reasoning=result.get("reasoning", ""),
        )
    except Exception as e:
        logger.warning(f"Claim verification failed for '{claim[:50]}...': {e}")
        return ClaimVerification(
            claim=claim,
            status="unsupported",
            reasoning=f"Verification failed: {e}",
        )


def critique(answer: str, context_text: str, query: str) -> CriticResult:
    """Run full critic evaluation on a generated answer.

    1. Decompose answer into claims
    2. Verify each claim against context
    3. Compute confidence score
    4. Produce grounded answer with flagged claims

    Args:
        answer: The generated answer.
        context_text: Formatted retrieval context.
        query: The original user query.

    Returns:
        CriticResult with confidence, claim verifications, and grounded answer.
    """
    logger.info("Critic: decomposing answer into claims...")
    claims = decompose_claims(answer)
    logger.info(f"Critic: found {len(claims)} claims to verify")

    if not claims:
        return CriticResult(
            confidence=0.5,
            grounded_answer=answer,
            feedback="Could not decompose answer into verifiable claims.",
        )

    # Verify each claim
    verifications = []
    for claim in claims:
        v = verify_claim(claim, context_text)
        verifications.append(v)

    # Count statuses
    supported = sum(1 for v in verifications if v.status == "supported")
    partial = sum(1 for v in verifications if v.status == "partially_supported")
    unsupported = sum(1 for v in verifications if v.status == "unsupported")
    total = len(verifications)

    # Confidence: supported=1.0, partial=0.5, unsupported=0.0
    confidence = (supported * 1.0 + partial * 0.5) / max(total, 1)

    # Build grounded answer — flag unsupported claims
    grounded_parts = []
    unsupported_claims = []
    for v in verifications:
        if v.status == "unsupported":
            unsupported_claims.append(v.claim)

    if unsupported_claims:
        grounded_answer = answer + "\n\n⚠️ **Note:** The following claims could not be verified against the source documents:\n"
        for claim in unsupported_claims:
            grounded_answer += f"- {claim}\n"
    else:
        grounded_answer = answer

    # Feedback for reformulation
    feedback_parts = []
    if unsupported > 0:
        feedback_parts.append(f"{unsupported} claims could not be verified against sources.")
    if partial > 0:
        feedback_parts.append(f"{partial} claims were only partially supported.")
    if confidence < 0.65:
        feedback_parts.append("Overall confidence is below threshold. Consider reformulating the query to retrieve more specific context.")

    feedback = " ".join(feedback_parts) if feedback_parts else "All claims verified."

    logger.info(
        f"Critic: confidence={confidence:.2f} "
        f"(supported={supported}, partial={partial}, unsupported={unsupported})"
    )

    return CriticResult(
        confidence=confidence,
        claims=verifications,
        grounded_answer=grounded_answer,
        feedback=feedback,
        supported_count=supported,
        partial_count=partial,
        unsupported_count=unsupported,
    )
