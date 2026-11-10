
import pytest

from src.orchestrator import build_retrieval_response


EVIDENCE = [
    {"text": "Positional encoding represents token order."},
    {"text": "Training used eight GPUs."},
]


@pytest.mark.parametrize(
    "route",
    ["vectorstore", "websearch"],
)
def test_successful_response_contract(route):
    result = build_retrieval_response(
        question="Why use positional encoding?",
        route=route,
        retrieval_result={
            "evidence": EVIDENCE,
            "relevant_evidence": [EVIDENCE[0]],
            "rejected_evidence": [EVIDENCE[1]],
            "agent_output": "It provides position information.",
            "error": None,
        },
    )

    assert result["route"] == route
    assert result["status"] == "retrieved"
    assert result["evidence"] == EVIDENCE
    assert result["relevant_evidence"] == [EVIDENCE[0]]
    assert result["rejected_evidence"] == [EVIDENCE[1]]
    assert result["agent_output"] is not None
    assert result["error"] is None


def test_no_evidence_status():
    result = build_retrieval_response(
        question="Unknown question",
        route="vectorstore",
        retrieval_result={
            "evidence": [],
            "relevant_evidence": [],
            "rejected_evidence": [],
            "agent_output": None,
            "error": None,
        },
    )

    assert result["status"] == "no_evidence"
    assert result["agent_output"] is None


def test_no_relevant_evidence_status():
    result = build_retrieval_response(
        question="Unrelated question",
        route="vectorstore",
        retrieval_result={
            "evidence": EVIDENCE,
            "relevant_evidence": [],
            "rejected_evidence": EVIDENCE,
            "agent_output": None,
            "error": None,
        },
    )

    assert result["status"] == "no_relevant_evidence"
    assert result["rejected_evidence"] == EVIDENCE


def test_grading_failure_status():
    result = build_retrieval_response(
        question="Test question",
        route="websearch",
        retrieval_result={
            "evidence": EVIDENCE,
            "relevant_evidence": [],
            "rejected_evidence": [],
            "agent_output": None,
            "error": {
                "type": "relevance_grading_error",
                "message": "Unable to grade retrieved evidence.",
            },
        },
    )

    assert result["status"] == "grading_failed"
    assert result["evidence"] == EVIDENCE
    assert result["agent_output"] is None


def test_generation_failure_status():
    result = build_retrieval_response(
        question="Test question",
        route="vectorstore",
        retrieval_result={
            "evidence": EVIDENCE,
            "relevant_evidence": EVIDENCE,
            "rejected_evidence": [],
            "agent_output": None,
            "error": {
                "type": "llm_generation_error",
                "message": "The LLM server failed.",
            },
        },
    )

    assert result["status"] == "generation_failed"
    assert result["evidence"] == EVIDENCE
    assert result["relevant_evidence"] == EVIDENCE