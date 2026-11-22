"""Liveness and bounded question execution."""

from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Depends, Request

from src.resilience import (
    ApplicationError, DependencyTimeout, DependencyUnavailable, ServiceBusy,
    check_deadline, request_scope,
)
from .dependencies import OrchestratorProvider, get_orchestrator
from .schemas import ErrorResponse, HealthResponse, QuestionRequest, QuestionResponse

router = APIRouter()
DOMAIN_ERRORS = {
    "grading_failed": ("relevance_grading_error", "Unable to grade retrieved evidence."),
    "generation_failed": ("llm_generation_error", "Unable to generate an answer."),
    "grounding_failed": ("grounding_validation_error", "Unable to validate generated answer."),
    "correction_failed": ("answer_correction_error", "Unable to complete answer correction."),
}


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


def execute_question(body, dependency, request):
    provider = request.app.state.orchestrator_provider
    with request_scope(request.state.request_id, provider.settings):
        orchestrator = None
        try:
            orchestrator = dependency.checkout() if isinstance(dependency, OrchestratorProvider) else dependency
            check_deadline()
            result = orchestrator.run(body.question)
            check_deadline()
            request.state.rag_route = result.get("route")
            request.state.pipeline_status = result.get("status")
            error = result.get("error")
            if error and error.get("code") == "DEPENDENCY_UNAVAILABLE":
                raise DependencyUnavailable()
            if error and error.get("code") == "DEPENDENCY_TIMEOUT":
                raise DependencyTimeout()
            # A generation error is emitted only for a recognized LLM server failure.
            if error and error.get("type") == "llm_generation_error":
                raise DependencyUnavailable()
            public_error = None
            if result["status"] in DOMAIN_ERRORS:
                kind, message = DOMAIN_ERRORS[result["status"]]
                public_error = dict(type=kind, code="PIPELINE_FAILED", message=message,
                                    request_id=request.state.request_id)
            elif error:
                raise ApplicationError()
            return QuestionResponse(
                question=result["question"], route=result["route"], status=result["status"],
                answer=result.get("verified_answer") if result["status"] == "verified" else None,
                sources=result.get("relevant_evidence") or [], error=public_error,
            )
        finally:
            if orchestrator is not None and isinstance(dependency, OrchestratorProvider):
                dependency.checkin(orchestrator)


@router.post("/api/v1/questions", response_model=QuestionResponse, responses={
    status: {"model": ErrorResponse} for status in (422, 500, 503, 504)
})
async def ask_question(
    body: QuestionRequest, request: Request,
    orchestrator: Annotated[Any, Depends(get_orchestrator)],
):
    provider = request.app.state.orchestrator_provider
    if not provider.slots.acquire(blocking=False):
        raise ServiceBusy()
    try:
        # Shield cancellation and wait for the blocking call to exit before releasing
        # admission. No custom executor, detached future, or background retry.
        with anyio.CancelScope(shield=True):
            return await anyio.to_thread.run_sync(
                execute_question, body, orchestrator, request, abandon_on_cancel=False,
            )
    finally:
        provider.slots.release()
