
from unittest.mock import Mock

import httpx
import pytest
from openai import InternalServerError

from src.orchestrator import (
    AgentOrchestrator,
    grade_and_summarize,
)
from src.relevance_grader import RelevanceGrader


EVIDENCE = [
    {
        "text": "Positional encoding represents token order.",
        "source": "sample.pdf",
        "page": 6,
    }
]


def make_server_error():
    request = httpx.Request(
        "POST",
        "http://localhost:11434/v1/chat/completions",
    )

    response = httpx.Response(
        status_code=500,
        request=request,
    )

    return InternalServerError(
        "Ollama prediction aborted",
        response=response,
        body=None,
    )


def test_invalid_grading_response_preserves_evidence():
    def invalid_scorer(question, evidence):
        raise ValueError("Invalid grading JSON")

    grader = RelevanceGrader(
        scorer=invalid_scorer
    )

    task_factory = Mock()

    result = grade_and_summarize(
        question="Why use positional encoding?",
        evidence=EVIDENCE,
        grader=grader,
        summary_agent=Mock(),
        task_factory=task_factory,
    )

    assert result["evidence"] == EVIDENCE
    assert result["relevant_evidence"] == []
    assert result["rejected_evidence"] == []
    assert result["agent_output"] is None
    assert result["error"]["type"] == (
        "relevance_grading_error"
    )

    task_factory.assert_not_called()


def test_ollama_failure_preserves_evidence():
    def failing_scorer(question, evidence):
        raise make_server_error()

    grader = RelevanceGrader(
        scorer=failing_scorer
    )

    task_factory = Mock()

    result = grade_and_summarize(
        question="Why use positional encoding?",
        evidence=EVIDENCE,
        grader=grader,
        summary_agent=Mock(),
        task_factory=task_factory,
    )

    assert result["evidence"] == EVIDENCE
    assert result["agent_output"] is None
    assert result["error"]["type"] == (
        "relevance_grading_error"
    )

    task_factory.assert_not_called()


@pytest.mark.parametrize(
    "route",
    ["vectorstore", "websearch"],
)
def test_both_routes_report_grading_failure(route):
    failed_result = {
        "evidence": EVIDENCE,
        "relevant_evidence": [],
        "rejected_evidence": [],
        "agent_output": None,
        "error": {
            "type": "relevance_grading_error",
            "message": "Unable to grade retrieved evidence.",
        },
    }

    orchestrator = AgentOrchestrator(
        router=lambda question: route,
        retriever=lambda question: failed_result,
        web_search=lambda question: failed_result,
    )

    result = orchestrator.run("Test question")

    assert result["route"] == route
    assert result["status"] == "grading_failed"
    assert result["evidence"] == EVIDENCE
    assert result["relevant_evidence"] == []
    assert result["rejected_evidence"] == []
    assert result["agent_output"] is None


def test_unexpected_grading_error_propagates():
    def failing_scorer(question, evidence):
        raise RuntimeError("Unexpected programming error")

    grader = RelevanceGrader(
        scorer=failing_scorer
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected programming error",
    ):
        grade_and_summarize(
            question="Test question",
            evidence=EVIDENCE,
            grader=grader,
            summary_agent=Mock(),
            task_factory=Mock(),
        )
