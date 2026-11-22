"""Deterministic resilience tests; no live Ollama, Tavily, or ChromaDB."""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
from threading import Event
from unittest.mock import Mock
from uuid import UUID, uuid4

from chromadb.errors import InternalError as ChromaInternalError
import httpx
from openai import APIConnectionError, APITimeoutError, InternalServerError
import pytest
import requests
from tavily.errors import TimeoutError as TavilyTimeout, InvalidAPIKeyError

from fastapi.testclient import TestClient
from src.api import create_app
from src.api.dependencies import get_orchestrator
from src.resilience import (
    DependencyTimeout, classify_exception, execution_context, llm_request_timeout,
    request_scope, stage_deadline,
)
from src.settings import Settings
from src.web_tools import WebSearchTool


QUESTION = {"question": "PRIVATE_PROMPT"}
RESULT = dict(question="PRIVATE_PROMPT", route="websearch", status="no_evidence",
              agent_output="UNTRUSTED", relevant_evidence=[], error=None)


@pytest.fixture
def api():
    app = create_app()
    orchestrator = Mock()
    orchestrator.run.return_value = RESULT.copy()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    with TestClient(app) as client:
        yield client, orchestrator


@pytest.mark.parametrize("exception,status,code", [
    (APIConnectionError(request=httpx.Request("POST", "http://localhost:11434")), 503, "DEPENDENCY_UNAVAILABLE"),
    (APITimeoutError(request=httpx.Request("POST", "http://localhost:11434")), 504, "DEPENDENCY_TIMEOUT"),
    (TavilyTimeout(15), 504, "DEPENDENCY_TIMEOUT"),
    (ChromaInternalError("PRIVATE_DATABASE"), 503, "DEPENDENCY_UNAVAILABLE"),
    (RuntimeError("PRIVATE_SECRET"), 500, "INTERNAL_ERROR"),
])
def test_dependency_errors_are_safe_and_slots_released(api, exception, status, code, caplog):
    client, orchestrator = api
    orchestrator.run.side_effect = exception
    request_id = str(uuid4())
    with caplog.at_level(logging.INFO):
        response = client.post("/api/v1/questions", json=QUESTION, headers={"X-Request-ID": request_id})
    assert response.status_code == status
    assert response.json()["code"] == code
    assert set(response.json()) == {"code", "message", "request_id"}
    assert response.json()["request_id"] == response.headers["X-Request-ID"] == request_id
    assert "PRIVATE" not in response.text
    assert "UNTRUSTED" not in response.text
    records = [r for r in caplog.records if getattr(r, "request_id", None) == request_id]
    assert any(r.getMessage() == "request_error" for r in records)
    completed = next(r for r in records if r.getMessage() == "request_completed")
    assert completed.http_route == "/api/v1/questions"
    assert completed.status_code == status
    assert completed.duration_ms >= 0
    assert "PRIVATE" not in caplog.text
    orchestrator.run.side_effect = None
    assert client.post("/api/v1/questions", json=QUESTION).status_code == 200


@pytest.mark.parametrize("inbound", [None, "unsafe-user-value", "x" * 1000, str(uuid4()).replace("-", "")])
def test_invalid_request_ids_are_replaced(api, inbound):
    client, _ = api
    headers = {"X-Request-ID": inbound} if inbound else {}
    response = client.get("/health", headers=headers)
    request_id = response.headers["X-Request-ID"]
    assert str(UUID(request_id)) == request_id
    assert request_id != inbound


def test_validation_never_echoes_input_or_invokes_orchestrator(api):
    client, orchestrator = api
    for body in ({"question": " \n\t"}, {"question": {"secret": "PRIVATE"}},
                 {"question": "x" * 10001}, {"question": "PRIVATE", "extra": "PRIVATE"}):
        response = client.post("/api/v1/questions", json=body)
        assert response.status_code == 422
        assert response.json()["code"] == "INVALID_REQUEST"
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
        assert "PRIVATE" not in response.text
    orchestrator.run.assert_not_called()


@pytest.mark.parametrize("status", ["verified", "unverified", "no_evidence", "no_relevant_evidence"])
def test_completed_outcomes_preserve_contract(api, status):
    client, orchestrator = api
    orchestrator.run.return_value = dict(RESULT, status=status, verified_answer="SAFE_VERIFIED")
    response = client.post("/api/v1/questions", json=QUESTION)
    assert response.status_code == 200
    assert response.json()["answer"] == ("SAFE_VERIFIED" if status == "verified" else None)
    assert "UNTRUSTED" not in response.text


@pytest.mark.parametrize("status", ["grading_failed", "generation_failed", "grounding_failed", "correction_failed"])
@pytest.mark.parametrize("code,http_status", [("DEPENDENCY_UNAVAILABLE", 503), ("DEPENDENCY_TIMEOUT", 504)])
def test_structured_infrastructure_failure_is_http_error(api, status, code, http_status):
    client, orchestrator = api
    orchestrator.run.return_value = dict(RESULT, status=status,
        verified_answer="MUST_NOT_ESCAPE", error={"code": code, "message": "PRIVATE"})
    response = client.post("/api/v1/questions", json=QUESTION)
    assert response.status_code == http_status
    assert response.json()["code"] == code
    assert "PRIVATE" not in response.text
    assert "MUST_NOT_ESCAPE" not in response.text


def test_real_pipeline_retains_domain_vs_server_failure_distinction(api, monkeypatch):
    from src.orchestrator import AgentOrchestrator, grade_and_summarize
    from src.relevance_grader import RelevanceGrader
    evidence = [{"text": "PRIVATE_DOCUMENT", "source": "sample.pdf"}]
    client, orchestrator = api

    def retrieve(_):
        return grade_and_summarize("question", evidence, grader, Mock(), Mock())

    runner = AgentOrchestrator(lambda _: "vectorstore", retrieve)
    orchestrator.run.side_effect = runner.run
    grader = RelevanceGrader(scorer=Mock(side_effect=ValueError("PRIVATE_PARSE")))
    response = client.post("/api/v1/questions", json=QUESTION)
    assert response.status_code == 200
    assert response.json()["status"] == "grading_failed"
    assert response.json()["error"]["code"] == "PIPELINE_FAILED"
    assert response.json()["sources"] == []
    assert "PRIVATE_PARSE" not in response.text

    error = InternalServerError("PRIVATE_OLLAMA", response=httpx.Response(500,
        request=httpx.Request("POST", "http://localhost:11434")), body=None)
    grader = RelevanceGrader(scorer=Mock(side_effect=error))
    response = client.post("/api/v1/questions", json=QUESTION)
    assert response.status_code == 503
    assert response.json()["code"] == "DEPENDENCY_UNAVAILABLE"
    assert "PRIVATE" not in response.text


def test_wrapped_ollama_timeout_is_still_timeout(api):
    client, orchestrator = api
    try:
        try:
            raise APITimeoutError(request=httpx.Request("POST", "http://localhost:11434"))
        except APITimeoutError as cause:
            raise ConnectionError("PRIVATE_WRAPPER") from cause
    except ConnectionError as exc:
        orchestrator.run.side_effect = exc
    response = client.post("/api/v1/questions", json=QUESTION)
    assert response.status_code == 504


def test_concurrent_overload_rejected_before_initialization(monkeypatch):
    started, release = Event(), Event()
    orchestrator = Mock()

    def blocked(_):
        started.set()
        assert release.wait(5)
        return RESULT.copy()

    orchestrator.run.side_effect = blocked
    factory = Mock(return_value=orchestrator)
    monkeypatch.setattr("src.api.dependencies.build_local_orchestrator", factory)
    with TestClient(create_app()) as client, ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/api/v1/questions", json=QUESTION)
        try:
            assert started.wait(5)
            second = client.post("/api/v1/questions", json=QUESTION)
            assert second.status_code == 503
            assert second.json()["code"] == "SERVICE_BUSY"
            assert client.get("/health").status_code == 200
            assert client.post("/api/v1/questions", json={"question": " "}).status_code == 422
            assert factory.call_count == 1
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200
        assert client.post("/api/v1/questions", json=QUESTION).status_code == 200
        assert factory.call_count == 1


def test_configured_concurrency_uses_separate_agents(monkeypatch):
    started, release = Event(), Event()
    calls = []

    def build():
        worker = Mock()
        def run(_):
            calls.append(worker)
            if len(calls) == 2:
                started.set()
            assert release.wait(5)
            return RESULT.copy()
        worker.run.side_effect = run
        return worker

    monkeypatch.setattr("src.api.dependencies.build_local_orchestrator", build)
    with TestClient(create_app(Settings(max_active=2))) as client, ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(client.post, "/api/v1/questions", json=QUESTION) for _ in range(2)]
        try:
            assert started.wait(5)
            assert calls[0] is not calls[1]
            assert client.post("/api/v1/questions", json=QUESTION).status_code == 503
        finally:
            release.set()
        assert all(f.result(timeout=5).status_code == 200 for f in pending)


def test_overall_deadline_does_not_abandon_worker(api):
    client, orchestrator = api
    def run(_):
        # Fake expiry avoids timing-sensitive sleeps.
        execution_context.get().deadline = 0
        return RESULT.copy()
    orchestrator.run.side_effect = run
    response = client.post("/api/v1/questions", json=QUESTION)
    assert response.status_code == 504
    orchestrator.run.side_effect = None
    assert client.post("/api/v1/questions", json=QUESTION).status_code == 200


def test_embedded_chroma_budget_checked_after_return(monkeypatch):
    clock = Mock(side_effect=[0.0, 11.0])
    monkeypatch.setattr("src.resilience.time.monotonic", clock)
    with pytest.raises(DependencyTimeout):
        with stage_deadline(10):
            pass


@pytest.fixture
def tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-placeholder")
    client = Mock()
    monkeypatch.setattr("src.web_tools.TavilyClient", Mock(return_value=client))
    monkeypatch.setattr("src.web_tools.time.sleep", Mock())
    return client


@pytest.mark.parametrize("transient", [TavilyTimeout(15), requests.exceptions.HTTPError(
    response=requests.Response())])
def test_tavily_attempts_capped_at_two(tavily, transient):
    if isinstance(transient, requests.exceptions.HTTPError):
        transient.response.status_code = 503
    tavily.search.side_effect = transient
    with pytest.raises(type(transient)):
        WebSearchTool()._run("question")
    assert tavily.search.call_count == 2
    tavily.close.assert_called_once()


def test_tavily_retry_success(tavily):
    tavily.search.side_effect = [TavilyTimeout(15), {"results": []}]
    assert json.loads(WebSearchTool()._run("question")) == []
    assert tavily.search.call_count == 2


@pytest.mark.parametrize("error", [ValueError("invalid"), InvalidAPIKeyError("invalid")])
def test_tavily_nontransient_not_retried(tavily, error):
    tavily.search.side_effect = error
    with pytest.raises(type(error)):
        WebSearchTool()._run("question")
    assert tavily.search.call_count == 1


def test_tavily_retry_respects_remaining_budget(tavily):
    tavily.search.side_effect = TavilyTimeout(15)
    with request_scope(str(uuid4()), Settings(request_timeout=1)):
        with pytest.raises(TavilyTimeout):
            WebSearchTool()._run("question")
    assert tavily.search.call_count == 1
    assert 0 < tavily.search.call_args.kwargs["timeout"] <= 1


def test_tavily_timeout_end_to_end(api, tavily):
    client, orchestrator = api
    tavily.search.side_effect = TavilyTimeout(15)
    orchestrator.run.side_effect = lambda _: WebSearchTool()._run("question")
    response = client.post("/api/v1/questions", json=QUESTION)
    assert response.status_code == 504
    assert response.json()["code"] == "DEPENDENCY_TIMEOUT"
    assert "answer" not in response.json()
    assert tavily.search.call_count == 2


def test_llm_httpx_hook_applies_real_deadline():
    captured = []
    def handle(request):
        captured.append(request.extensions["timeout"])
        return httpx.Response(200, json={})
    with request_scope(str(uuid4()), Settings(llm_timeout=60, request_timeout=2)):
        with httpx.Client(transport=httpx.MockTransport(handle),
                          event_hooks={"request": [llm_request_timeout]}) as client:
            client.post("http://localhost:11434/v1/chat/completions", json={})
        assert 0 < captured[0]["read"] <= 2
        execution_context.get().deadline = 0
        with pytest.raises(DependencyTimeout):
            llm_request_timeout(httpx.Request("POST", "http://localhost:11434"))


def test_installed_crewai_uses_configured_timeout_and_no_retries():
    from src.agents import create_router_agent
    agent = create_router_agent()
    assert agent.verbose is False
    assert agent.max_retry_limit == 0
    llm = agent.llm
    assert llm.timeout == 60
    assert llm.max_retries == 0
    # Verify actual installed SDK configuration, not just constructor kwargs.
    client = llm._get_sync_client()
    assert client.max_retries == 0
    assert client.timeout == 60
    from src.llm_transport import DeadlineInterceptor
    assert isinstance(client._client._transport.interceptor, DeadlineInterceptor)
    client.close()


@pytest.mark.parametrize("kwargs", [dict(llm_timeout=0), dict(request_timeout=float("inf")),
    dict(chroma_timeout=-1), dict(max_active=0), dict(tavily_timeout=float("nan"))])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_environment_settings(monkeypatch):
    monkeypatch.setenv("TAVILY_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("CHROMA_TIMEOUT_SECONDS", "4")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("RAG_TIMEOUT_SECONDS", "6")
    monkeypatch.setenv("RAG_MAX_ACTIVE_REQUESTS", "2")
    assert Settings.from_env() == Settings(3, 4, 5, 6, 2)


def test_client_cancellation_does_not_release_running_worker():
    import anyio
    from starlette.requests import Request
    from src.api.routes import ask_question
    from src.api.schemas import QuestionRequest

    started, release = Event(), Event()
    app = create_app()
    provider = app.state.orchestrator_provider
    orchestrator = Mock()
    def blocked(_):
        started.set()
        assert release.wait(5)
        return RESULT.copy()
    orchestrator.run.side_effect = blocked
    request = Request({"type": "http", "app": app, "state": {"request_id": str(uuid4())}})

    async def scenario():
        scope = anyio.CancelScope()
        async def call():
            with scope:
                await ask_question(QuestionRequest(**QUESTION), request, orchestrator)
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(call)
            try:
                assert await anyio.to_thread.run_sync(started.wait, 5)
                scope.cancel()
                await anyio.sleep(0)
                # Cancellation must not turn an occupied slot into free capacity.
                assert not provider.slots.acquire(blocking=False)
            finally:
                release.set()
        assert provider.slots.acquire(blocking=False)
        provider.slots.release()

    anyio.run(scenario)


def test_vector_store_filesystem_failure_is_dependency_unavailable(api, monkeypatch):
    from src.vector_store import VectorStore
    client, orchestrator = api
    monkeypatch.setattr("src.vector_store.SentenceTransformer", Mock(side_effect=OSError("PRIVATE_PATH")))
    orchestrator.run.side_effect = lambda _: VectorStore()
    response = client.post("/api/v1/questions", json=QUESTION)
    assert response.status_code == 503
    assert response.json()["code"] == "DEPENDENCY_UNAVAILABLE"
    assert "PRIVATE_PATH" not in response.text
