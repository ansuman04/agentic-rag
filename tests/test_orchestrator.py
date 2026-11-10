
import pytest

from src.orchestrator import AgentOrchestrator


def test_vectorstore_route():
    calls = []

    expected_evidence = [
        {
            "text": "Positional encoding provides order information.",
            "source": "sample.pdf",
            "page": 6,
            "chunk_index": 2,
            "document_id": "document-123",
            "distance": 0.12,
        }
    ]

    def router(question):
        calls.append("router")
        return "vectorstore"

    def retriever(question):
        calls.append("retriever")
        return {
            "evidence": expected_evidence,
            "agent_output": "Positional encoding represents position.",
        }

    orchestrator = AgentOrchestrator(
        router=router,
        retriever=retriever,
    )

    result = orchestrator.run(
        "Why do Transformers use positional encoding?"
    )

    assert result["route"] == "vectorstore"
    assert result["status"] == "retrieved"
    assert result["evidence"] == expected_evidence
    assert result["agent_output"] == (
        "Positional encoding represents position."
    )

    assert calls == ["router", "retriever"]

def test_websearch_route_does_not_call_retriever():
    def retriever(question):
        raise AssertionError(
            "Document retriever must not run for websearch."
        )

    orchestrator = AgentOrchestrator(
        router=lambda question: "websearch",
        retriever=retriever,
    )

    result = orchestrator.run(
        "What is the weather today?"
    )

    assert result["route"] == "websearch"
    assert result["evidence"] is None
    assert "not implemented" in result["message"].lower()


def test_invalid_route():
    orchestrator = AgentOrchestrator(
        router=lambda question: "unknown",
        retriever=lambda question: "unused",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported routing decision",
    ):
        orchestrator.run("Example question")


def test_empty_question():
    orchestrator = AgentOrchestrator(
        router=lambda question: "vectorstore",
        retriever=lambda question: "unused",
    )

    with pytest.raises(ValueError):
        orchestrator.run("   ")


def test_websearch_route_calls_web_agent():
    calls = []

    expected_evidence = [
        {
            "title": "Python Releases",
            "url": "https://www.python.org/",
            "snippet": "Python release information.",
        }
    ]

    def router(question):
        calls.append("router")
        return "websearch"

    def retriever(question):
        raise AssertionError(
            "Document retriever should not run."
        )

    def web_search(question):
        calls.append("websearch")
        return {
            "evidence": expected_evidence,
            "agent_output": "Python release summary.",
        }

    orchestrator = AgentOrchestrator(
        router=router,
        retriever=retriever,
        web_search=web_search,
    )

    result = orchestrator.run(
        "What is the latest Python release?"
    )

    assert result["route"] == "websearch"
    assert result["status"] == "retrieved"
    assert result["evidence"] == expected_evidence
    assert result["agent_output"] == (
        "Python release summary."
    )
    assert calls == ["router", "websearch"]