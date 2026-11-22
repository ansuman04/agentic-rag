"""Application factory, sanitized errors and request correlation."""

import logging
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from src.resilience import ApplicationError, classify_exception, valid_request_id
from src.settings import Settings
from .dependencies import OrchestratorProvider
from .routes import router

logger = logging.getLogger(__name__)


def error_response(request: Request, code: str, message: str, status_code: int):
    return JSONResponse(status_code=status_code, content={
        "code": code, "message": message, "request_id": request.state.request_id,
    })


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Agentic RAG API", version="0.2.0")
    app.state.orchestrator_provider = OrchestratorProvider(settings)
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI's default error includes the invalid input; never echo it.
        return error_response(request, "INVALID_REQUEST", "The request is invalid.", 422)

    @app.exception_handler(ApplicationError)
    async def application_error(request, exc):
        logger.warning("request_error", extra={"request_id": request.state.request_id,
                       "error_code": exc.code, "exception_type": type(exc).__name__})
        return error_response(request, exc.code, exc.public_message, exc.status_code)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error_response(request, "HTTP_ERROR", "The HTTP request could not be completed.", exc.status_code)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request.state.request_id = valid_request_id(request.headers.get("X-Request-ID"))
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as exc:
            error = classify_exception(exc)
            cause = exc.__cause__ or exc.__context__
            logger.error("request_error", extra={
                "request_id": request.state.request_id, "error_code": error.code,
                "exception_type": type(exc).__name__,
                "cause_type": type(cause).__name__ if cause else None,
            })
            response = error_response(request, error.code, error.public_message, error.status_code)
        response.headers["X-Request-ID"] = request.state.request_id
        route = request.scope.get("route")
        logger.info("request_completed", extra={
            "request_id": request.state.request_id,
            "http_route": getattr(route, "path", "unmatched"),
            "rag_route": getattr(request.state, "rag_route", None),
            "pipeline_status": getattr(request.state, "pipeline_status", None),
            "status_code": response.status_code,
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        })
        return response

    return app


app = create_app()
