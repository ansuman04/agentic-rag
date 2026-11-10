
from unittest.mock import Mock, patch

import pytest

from src.relevance_grader import RelevanceGrader
from src.orchestrator import grade_and_summarize


EVIDENCE = [
    {"text": "Positional encoding represents order."},
    {"text": "Training used eight GPUs."},
]


def test_only_relevant_evidence_is_summarized():
    grader = RelevanceGrader(
        scorer=lambda question, items: [True, False]
    )

    task_factory = Mock(return_value=Mock())

    with patch(
        "src.orchestrator.run_summary_safely",
        return_value={
            "agent_output": "Positions encode order.",
            "error": None,
        },
    ) as summarize:
        result = grade_and_summarize(
            question="Why use positional encoding?",
            evidence=EVIDENCE,
            grader=grader,
            summary_agent=Mock(),
            task_factory=task_factory,
        )

    assert result["evidence"] == EVIDENCE
    assert result["relevant_evidence"] == [EVIDENCE[0]]
    assert result["rejected_evidence"] == [EVIDENCE[1]]

    assert task_factory.call_args.kwargs[
        "evidence"
    ] == [EVIDENCE[0]]

    assert summarize.call_count == 1


def test_no_relevant_evidence_skips_summary():
    grader = RelevanceGrader(
        scorer=lambda question, items: [False, False]
    )

    task_factory = Mock()

    result = grade_and_summarize(
        question="Unrelated question",
        evidence=EVIDENCE,
        grader=grader,
        summary_agent=Mock(),
        task_factory=task_factory,
    )

    assert result["relevant_evidence"] == []
    assert result["rejected_evidence"] == EVIDENCE
    assert result["agent_output"] is None

    task_factory.assert_not_called()


def test_empty_evidence_skips_grading():
    grader = RelevanceGrader(
        scorer=lambda question, items: pytest.fail(
            "Scorer should not run"
        )
    )

    result = grade_and_summarize(
        question="Test",
        evidence=[],
        grader=grader,
        summary_agent=Mock(),
        task_factory=Mock(),
    )

    assert result["evidence"] == []
    assert result["agent_output"] is None


def test_grading_failure_returns_structured_error():
    def failing_scorer(question, evidence):
        raise ValueError("Invalid grading response")

    grader = RelevanceGrader(
        scorer=failing_scorer
    )

    task_factory = Mock()

    result = grade_and_summarize(
        question="Test question",
        evidence=EVIDENCE,
        grader=grader,
        summary_agent=Mock(),
        task_factory=task_factory,
    )

    # Original retrieved evidence must be preserved.
    assert result["evidence"] == EVIDENCE

    # No evidence was successfully graded.
    assert result["relevant_evidence"] == []
    assert result["rejected_evidence"] == []

    # Summarization must not run after grading fails.
    assert result["agent_output"] is None
    task_factory.assert_not_called()

    # The failure must be explicitly reported.
    assert result["error"] == {
        "type": "relevance_grading_error",
        "message": "Unable to grade retrieved evidence.",
    }
