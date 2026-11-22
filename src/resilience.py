"""Shared error taxonomy and cooperative deadline context for blocking stages."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import time
from uuid import UUID, uuid4

from src.settings import Settings

logger = logging.getLogger(__name__)


class ApplicationError(RuntimeError):
    code = "INTERNAL_ERROR"
    status_code = 500
    public_message = "The request could not be completed."


class DependencyUnavailable(ApplicationError):
    code = "DEPENDENCY_UNAVAILABLE"
    status_code = 503
    public_message = "A required dependency is unavailable."


class DependencyTimeout(ApplicationError):
    code = "DEPENDENCY_TIMEOUT"
    status_code = 504
    public_message = "A dependency or request deadline expired."


class ServiceBusy(ApplicationError):
    code = "SERVICE_BUSY"
    status_code = 503
    public_message = "The service is busy. Try again later."


@dataclass
class ExecutionContext:
    request_id: str
    deadline: float
    settings: Settings


execution_context: ContextVar[ExecutionContext | None] = ContextVar("rag_execution", default=None)


def valid_request_id(value: str | None) -> str:
    if value and len(value) == 36:
        try:
            parsed = UUID(value)
            if str(parsed) == value.lower():
                return str(parsed)
        except ValueError:
            pass
    return str(uuid4())


@contextmanager
def request_scope(request_id: str, settings: Settings):
    token = execution_context.set(ExecutionContext(
        request_id=request_id, deadline=time.monotonic() + settings.request_timeout, settings=settings,
    ))
    try:
        yield
    finally:
        execution_context.reset(token)


def current_settings() -> Settings:
    context = execution_context.get()
    return context.settings if context else Settings.from_env()


def remaining_budget() -> float:
    context = execution_context.get()
    return max(0.0, context.deadline - time.monotonic()) if context else float("inf")


def check_deadline() -> None:
    if remaining_budget() <= 0:
        raise DependencyTimeout()


def stage_timeout(configured: float) -> float:
    check_deadline()
    return min(configured, remaining_budget())


@contextmanager
def stage_deadline(seconds: float):
    """Embedded operations cannot be interrupted; check elapsed time on return."""
    started = time.monotonic()
    check_deadline()
    yield
    if time.monotonic() - started >= seconds:
        raise DependencyTimeout()
    check_deadline()


def classify_exception(exc: Exception) -> ApplicationError:
    # Inspect types and explicit chains, never substrings in provider messages.
    import httpx
    import openai
    import requests
    from chromadb.errors import ChromaError
    from tavily import errors as tavily_errors
    import sqlite3

    chain = []
    current = exc
    while current is not None and id(current) not in {id(item) for item in chain}:
        chain.append(current)
        current = current.__cause__ or current.__context__
    for item in chain:
        if isinstance(item, ApplicationError):
            return item
        if isinstance(item, (openai.APITimeoutError, httpx.TimeoutException,
                             requests.exceptions.Timeout, tavily_errors.TimeoutError, TimeoutError)):
            return DependencyTimeout()
    for item in chain:
        if isinstance(item, (openai.APIConnectionError, ConnectionError, httpx.NetworkError,
                             requests.exceptions.ConnectionError, sqlite3.OperationalError,
                             tavily_errors.InvalidAPIKeyError, tavily_errors.MissingAPIKeyError,
                             tavily_errors.ForbiddenError, tavily_errors.UsageLimitExceededError)):
            return DependencyUnavailable()
        if isinstance(item, ChromaError) and item.code() in (401, 403, 429, 500, 503):
            return DependencyUnavailable()
        if isinstance(item, (openai.APIStatusError, httpx.HTTPStatusError, requests.exceptions.HTTPError)):
            response = getattr(item, "response", None)
            status = getattr(item, "status_code", None) or getattr(response, "status_code", None)
            if status in (408, 504):
                return DependencyTimeout()
            if status in (401, 403, 404, 429, 500, 502, 503):
                return DependencyUnavailable()
    return ApplicationError()


def log_failure(exc: Exception, stage: str):
    context = execution_context.get()
    cause = exc.__cause__ or exc.__context__
    # Exception bodies and stack locals can contain prompts, credentials, or evidence.
    logger.warning("dependency_failure", extra={
        "request_id": context.request_id if context else None,
        "stage": stage, "exception_type": type(exc).__name__,
        "cause_type": type(cause).__name__ if cause else None,
        "error_code": classify_exception(exc).code,
    })


def pipeline_error(kind: str, message: str, exc: Exception) -> dict:
    log_failure(exc, kind)
    result = {"type": kind, "message": message}
    error = classify_exception(exc)
    if error.code != "INTERNAL_ERROR":
        result["code"] = error.code
    return result


def llm_request_timeout(request):
    """HTTPX request hook: real socket deadlines, clipped before each LLM call."""
    timeout = stage_timeout(current_settings().llm_timeout)
    request.extensions["timeout"] = dict(connect=timeout, read=timeout, write=timeout, pool=timeout)
