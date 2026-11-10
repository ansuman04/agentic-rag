
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import InternalServerError

from src.orchestrator import (
    AgentOrchestrator,
    run_summary_safely,
)


EVIDENCE = [
    {
        "title": "Example Source",
        "url": "https://example.com",
        "snippet": "Original evidence.",
    }
]


def make_llm_server_error():
    request = httpx.Request(
        "POST",
        "http://localhost:11434/v1/chat/completions",
    )
    response = httpx.Response(
        status_code=500,
        request=request,
    )

    return InternalServerError(
        "prediction aborted, token repeat limit reached",
        response=response,
        body=None,
    )


def test_successful_summary_preserves_evidence():
    with patch(
        "src.orchestrator.run_agent",
        return_value="Generated summary.",
    ):
        result = run_summary_safely(
            agent=Mock(),
            task=Mock(),
            evidence=EVIDENCE,
        )

    assert result["evidence"] == EVIDENCE
    assert result["agent_output"] == "Generated summary."
    assert result["error"] is None


def test_llm_failure_preserves_evidence():
    with patch(
        "src.orchestrator.run_agent",
        side_effect=make_llm_server_error(),
    ):
        result = run_summary_safely(
            agent=Mock(),
            task=Mock(),
            evidence=EVIDENCE,
        )

    assert result["evidence"] == EVIDENCE
    assert result["agent_output"] is None
    assert result["error"]["type"] == (
        "llm_generation_error"
    )


def test_unexpected_exception_is_not_hidden():
    with patch(
        "src.orchestrator.run_agent",
        side_effect=ValueError("Programming error"),
    ):
        with pytest.raises(
            ValueError,
            match="Programming error",
        ):
            run_summary_safely(
                agent=Mock(),
                task=Mock(),
                evidence=EVIDENCE,
            )


@pytest.mark.parametrize(
    "route",
    ["vectorstore", "websearch"],
)
def test_both_routes_report_generation_failure(route):
    def failed_summary(_question):
        return {
            "evidence": EVIDENCE,
            "agent_output": None,
            "error": {
                "type": "llm_generation_error",
                "message": "The LLM server failed.",
            },
        }

    orchestrator = AgentOrchestrator(
        router=lambda _: route,
        retriever=failed_summary,
        web_search=failed_summary,
    )

    result = orchestrator.run("Test question")

    assert result["route"] == route
    assert result["status"] == "generation_failed"
    assert result["evidence"] == EVIDENCE
    assert result["agent_output"] is None
    assert result["error"]["type"] == (
        "llm_generation_error"
    )
