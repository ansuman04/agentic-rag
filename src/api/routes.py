"""P0.1 liveness and question endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from .dependencies import get_orchestrator
from .schemas import HealthResponse, QuestionRequest, QuestionResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.post("/api/v1/questions", response_model=QuestionResponse)
def ask_question(
    body: QuestionRequest,
    request: Request,
    orchestrator: Annotated[Any, Depends(get_orchestrator)],
):
    # CrewAI agents are shared and mutable; serialize calls within this process.
    with request.app.state.orchestrator_provider.lock:
        result = orchestrator.run(body.question)
    return QuestionResponse(
        question=result["question"],
        route=result["route"],
        status=result["status"],
        answer=result.get("verified_answer") if result["status"] == "verified" else None,
        sources=result.get("relevant_evidence") or [],
        error=result.get("error"),
    )
