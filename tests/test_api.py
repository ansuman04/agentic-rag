from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.api import app, get_orchestrator


def setup_function():
    app.dependency_overrides.clear()


def teardown_function():
    app.dependency_overrides.clear()


def test_health_does_not_initialize_orchestrator():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_verified_answer_is_exposed():
    orchestrator = Mock()
    orchestrator.run.return_value = {
        "question": "Why positional encoding?",
        "route": "vectorstore",
        "status": "verified",
        "verified_answer": "It represents token positions.",
        "agent_output": "Untrusted draft.",
        "relevant_evidence": [{"text": "Position information", "source": "paper.pdf"}],
        "error": None,
    }
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    response = TestClient(app).post(
        "/api/v1/questions", json={"question": " Why positional encoding? "}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "It represents token positions."
    assert data["status"] == "verified"
    assert data["sources"][0]["source"] == "paper.pdf"
    assert "agent_output" not in data
    orchestrator.run.assert_called_once_with("Why positional encoding?")


def test_unverified_answer_is_not_exposed():
    orchestrator = Mock()
    orchestrator.run.return_value = {
        "route": "websearch",
        "status": "unverified",
        "verified_answer": None,
        "agent_output": "Potentially hallucinated draft.",
        "relevant_evidence": [],
        "error": None,
    }
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    response = TestClient(app).post(
        "/api/v1/questions", json={"question": "What happened?"}
    )
    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert response.json()["status"] == "unverified"


def test_inconsistent_verified_status_fails_closed():
    orchestrator = Mock()
    orchestrator.run.return_value = {
        "route": "vectorstore",
        "status": "verified",
        "verified_answer": None,
        "relevant_evidence": [],
        "error": None,
    }
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    response = TestClient(app).post(
        "/api/v1/questions", json={"question": "Example"}
    )
    assert response.json()["status"] == "unverified"
    assert response.json()["answer"] is None


def test_blank_question_is_rejected():
    response = TestClient(app).post(
        "/api/v1/questions", json={"question": "   "}
    )
    assert response.status_code == 422


def test_missing_question_is_rejected():
    response = TestClient(app).post("/api/v1/questions", json={})
    assert response.status_code == 422


def test_service_failure_is_sanitized():
    orchestrator = Mock()
    orchestrator.run.side_effect = RuntimeError("secret internal detail")
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    response = TestClient(app).post(
        "/api/v1/questions", json={"question": "Example"}
    )
    assert response.status_code == 503
    assert "secret internal detail" not in response.text


def test_structured_pipeline_failure_is_preserved():
    orchestrator = Mock()
    orchestrator.run.return_value = {
        "route": "websearch",
        "status": "grading_failed",
        "verified_answer": None,
        "relevant_evidence": [{"url": "https://example.com", "snippet": "Text"}],
        "error": {"type": "relevance_grading_error", "message": "Unable to grade."},
    }
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    response = TestClient(app).post(
        "/api/v1/questions", json={"question": "Example"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "grading_failed"
    assert response.json()["answer"] is None
    assert response.json()["error"]["type"] == "relevance_grading_error"
