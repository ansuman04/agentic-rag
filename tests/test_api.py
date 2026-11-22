from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.dependencies import OrchestratorProvider, get_orchestrator


@pytest.fixture
def api():
    app = create_app()
    orchestrator = Mock()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, orchestrator


def test_health_and_schema_do_not_initialize(monkeypatch):
    factory = Mock(side_effect=AssertionError("must stay lazy"))
    monkeypatch.setattr("src.api.dependencies.build_local_orchestrator", factory)
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths) == {"/health", "/api/v1/questions"}
        assert client.post("/api/v1/questions", json={"question": " "}).status_code == 422
    factory.assert_not_called()


@pytest.mark.parametrize("body", [None, {}, {"question": ""}, {"question": " \n\t"},
    {"question": None}, {"question": 42}, {"question": True}, {"question": []},
    {"question": "x" * 10001}, {"question": "x", "route": "websearch"}])
def test_invalid_question(api, body):
    client, orchestrator = api
    assert client.post("/api/v1/questions", json=body).status_code == 422
    orchestrator.run.assert_not_called()


@pytest.mark.parametrize("route,sources", [
    ("vectorstore", [{"text": "Evidence", "source": "sample.pdf", "page": 2,
                      "chunk_index": 0, "document_id": "abc", "distance": 0.2}]),
    ("websearch", [{"title": "Source", "url": "https://example.com", "snippet": "Evidence"}]),
])
@pytest.mark.parametrize("status", ["verified", "unverified", "no_evidence",
    "no_relevant_evidence", "grading_failed", "generation_failed",
    "grounding_failed", "correction_failed"])
def test_public_response(api, route, sources, status):
    client, orchestrator = api
    error = {"type": "example_error", "message": "Pipeline failure"} if status.endswith("failed") else None
    orchestrator.run.return_value = dict(
        question="hello", route=route, status=status, relevant_evidence=sources,
        evidence=sources + [{"text": "rejected"}], rejected_evidence=[{"text": "rejected"}],
        agent_output="UNTRUSTED", original_answer="UNTRUSTED",
        verified_answer="Verified text", error=error,
    )
    response = client.post("/api/v1/questions", json={"question": " hello "})
    assert response.status_code == 200
    assert response.json() == dict(question="hello", route=route, status=status,
        answer="Verified text" if status == "verified" else None, sources=sources, error=error)
    assert "UNTRUSTED" not in response.text
    orchestrator.run.assert_called_once_with("hello")


def test_verified_without_verified_answer_never_falls_back(api):
    client, orchestrator = api
    orchestrator.run.return_value = dict(question="hello", route="vectorstore",
        status="verified", agent_output="UNTRUSTED", evidence=[{"text": "ungraded"}])
    response = client.post("/api/v1/questions", json={"question": "hello"})
    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert response.json()["sources"] == []


def test_unexpected_error_is_generic(api):
    client, orchestrator = api
    orchestrator.run.side_effect = RuntimeError("private credentials")
    response = client.post("/api/v1/questions", json={"question": "hello"})
    assert response.status_code == 500
    assert response.json() == {"detail": "RAG processing failed. Check server logs."}


def test_real_dependency_reuses_factory_across_requests(monkeypatch):
    orchestrator = Mock()
    orchestrator.run.return_value = dict(question="hello", route="vectorstore",
        status="no_evidence", relevant_evidence=[], verified_answer=None, error=None)
    factory = Mock(return_value=orchestrator)
    monkeypatch.setattr("src.api.dependencies.build_local_orchestrator", factory)
    with TestClient(create_app()) as client:
        for _ in range(2):
            assert client.post("/api/v1/questions", json={"question": "hello"}).status_code == 200
    factory.assert_called_once_with()
    assert orchestrator.run.call_count == 2


def test_concurrent_initialization_builds_once(monkeypatch):
    orchestrator = Mock()
    factory = Mock(return_value=orchestrator)
    monkeypatch.setattr("src.api.dependencies.build_local_orchestrator", factory)
    provider = OrchestratorProvider()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: provider.get(), range(32)))
    assert all(result is orchestrator for result in results)
    factory.assert_called_once_with()


def test_initialization_failure_returns_500_and_can_retry(monkeypatch):
    orchestrator = Mock()
    orchestrator.run.return_value = dict(question="hello", route="websearch", status="no_evidence")
    factory = Mock(side_effect=[RuntimeError("private configuration"), orchestrator])
    monkeypatch.setattr("src.api.dependencies.build_local_orchestrator", factory)
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post("/api/v1/questions", json={"question": "hello"})
        assert response.status_code == 500
        assert "private configuration" not in response.text
        assert client.get("/health").status_code == 200
        assert client.post("/api/v1/questions", json={"question": "hello"}).status_code == 200
    assert factory.call_count == 2
