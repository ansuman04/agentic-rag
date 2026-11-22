"""HTTP interface for the existing Agentic RAG orchestrator.

The API returns only grounded answers as public answers. The underlying
orchestrator is loaded lazily to avoid loading local models on import.
"""
from functools import lru_cache
import logging
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.orchestrator import AgentOrchestrator, build_local_orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG API",
    version="0.1.0",
    description="Evidence-grounded document and web question answering.",
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class EvidenceItem(BaseModel):
    model_config = {"extra": "allow"}


class QuestionResponse(BaseModel):
    question: str
    route: str
    status: str
    answer: str | None
    sources: list[dict[str, Any]]
    error: dict[str, Any] | None = None


@lru_cache(maxsize=1)
def get_orchestrator() -> AgentOrchestrator:
    """Create the heavyweight CrewAI/ChromaDB services once per process."""
    return build_local_orchestrator()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only; does not assert Ollama or Tavily availability."""
    return {"status": "ok"}


@app.post("/api/v1/questions", response_model=QuestionResponse)
def ask_question(
    request: QuestionRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> QuestionResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")

    try:
        result = orchestrator.run(question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Agentic RAG request failed")
        raise HTTPException(
            status_code=503,
            detail="The question answering service is temporarily unavailable.",
        ) from None

    status = result.get("status", "unverified")
    verified_answer = result.get("verified_answer")
    # Fail closed if orchestration produced an inconsistent response.
    answer = (
        verified_answer
        if status == "verified" and isinstance(verified_answer, str)
        and verified_answer.strip()
        else None
    )
    if status == "verified" and answer is None:
        status = "unverified"

    evidence = result.get("relevant_evidence") or []
    return QuestionResponse(
        question=question,
        route=str(result.get("route", "unknown")),
        status=status,
        answer=answer,
        sources=evidence if isinstance(evidence, list) else [],
        error=result.get("error"),
    )
