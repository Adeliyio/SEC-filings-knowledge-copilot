import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin import router as admin_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.eval import router as eval_router
from app.api.feedback import router as feedback_router
from app.api.search import router as search_router
from app.config import settings

logger = logging.getLogger(__name__)


def _warmup_ollama_models() -> dict[str, bool]:
    """Pre-pull and warm up Ollama models so first query isn't cold."""
    results = {}
    for model in [settings.ollama_model, settings.ollama_embed_model]:
        try:
            # Pull model (no-op if already present, downloads if missing)
            logger.info(f"Warming up Ollama model: {model}")
            httpx.post(
                f"{settings.ollama_host}/api/pull",
                json={"name": model, "stream": False},
                timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
            )
            # Run a tiny inference to load model into memory
            if model == settings.ollama_embed_model:
                httpx.post(
                    f"{settings.ollama_host}/api/embed",
                    json={"model": model, "input": ["warmup"]},
                    timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
                )
            else:
                httpx.post(
                    f"{settings.ollama_host}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                    timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
                )
            results[model] = True
            logger.info(f"Model {model} warmed up successfully")
        except Exception as e:
            results[model] = False
            logger.warning(f"Failed to warm up model {model}: {e}")
    return results


def _warmup_reranker() -> bool:
    """Pre-load the cross-encoder reranker model."""
    try:
        from app.storage.search import _get_reranker
        logger.info("Warming up cross-encoder reranker...")
        _get_reranker()
        logger.info("Cross-encoder reranker loaded")
        return True
    except Exception as e:
        logger.warning(f"Failed to warm up reranker: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-warm models so first user query is fast."""
    logger.info("Starting model warmup...")
    ollama_status = _warmup_ollama_models()
    reranker_ok = _warmup_reranker()
    if all(ollama_status.values()) and reranker_ok:
        logger.info("All models warmed up — ready to serve queries")
    else:
        logger.warning(
            f"Some models failed warmup: ollama={ollama_status}, reranker={reranker_ok}. "
            "First queries may be slow."
        )
    yield


app = FastAPI(
    title="Enterprise Knowledge Copilot",
    description="AI copilot for SEC 10-K financial filings with multi-agent self-correction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.include_router(chat_router)
app.include_router(feedback_router)
app.include_router(eval_router)
app.include_router(documents_router)
app.include_router(admin_router)


@app.get("/health")
async def health():
    """Health check with dependency status."""
    checks = {"status": "ok"}

    # Check Ollama
    try:
        resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        checks["ollama"] = "ok"
    except Exception:
        checks["ollama"] = "unavailable"
        checks["status"] = "degraded"

    # Check Qdrant
    try:
        from app.storage.qdrant import get_client
        client = get_client()
        client.get_collections()
        checks["qdrant"] = "ok"
    except Exception:
        checks["qdrant"] = "unavailable"
        checks["status"] = "degraded"

    # Check PostgreSQL
    try:
        from sqlalchemy import text
        from app.storage.postgres import get_sync_session_factory
        session = get_sync_session_factory()()
        try:
            session.execute(text("SELECT 1"))
            checks["postgres"] = "ok"
        finally:
            session.close()
    except Exception:
        checks["postgres"] = "unavailable"
        checks["status"] = "degraded"

    return checks
