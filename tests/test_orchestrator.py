
import pytest

from src.orchestrator import AgentOrchestrator


def test_vectorstore_route():
    calls = []

    def router(question):
        calls.append("router")
        return "vectorstore"

    def retriever(question):
        calls.append("retriever")
        return "Evidence from sample.pdf, page 6"

    orchestrator = AgentOrchestrator(
        router=router,
        retriever=retriever,
    )

    result = orchestrator.run(
        "Why do Transformers use positional encoding?"
    )

    assert result["route"] == "vectorstore"
    assert "sample.pdf" in result["evidence"]
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

    def router(question):
        calls.append("router")
        return "websearch"

    def retriever(question):
        raise AssertionError(
            "Document retriever should not run."
        )

    def web_search(question):
        calls.append("websearch")
        return (
            "Python release information from "
            "https://www.python.org/"
        )

    orchestrator = AgentOrchestrator(
        router=router,
        retriever=retriever,
        web_search=web_search,
    )

    result = orchestrator.run(
        "What is the latest Python release?"
    )

    assert result["route"] == "websearch"
    assert "python.org" in result["evidence"]
    assert calls == ["router", "websearch"]