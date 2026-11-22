
from unittest.mock import Mock

from src.orchestrator import validate_generated_answer


EVIDENCE = [
    {
        "text": "Transformers were introduced in 2017.",
        "source": "sample.pdf",
        "page": 1,
    }
]


def make_result():
    return {
        "evidence": EVIDENCE,
        "relevant_evidence": EVIDENCE,
        "rejected_evidence": [],
        "agent_output": "Transformers were introduced in 2020.",
        "error": None,
    }


def test_grounding_runs_after_successful_summary():
    validator = Mock()

    validator.validate.return_value = {
        "grounding_status": "unsupported",
        "claims": [
            {
                "claim": "Transformers were introduced in 2020.",
                "verdict": "unsupported",
                "reason": "Evidence states 2017.",
            }
        ],
    }

    result = validate_generated_answer(
        question="When were Transformers introduced?",
        retrieval_result=make_result(),
        grounding_validator=validator,
    )

    assert result["grounding"]["grounding_status"] == (
        "unsupported"
    )

    assert result["agent_output"] == (
        "Transformers were introduced in 2020."
    )

    validator.validate.assert_called_once_with(
        question="When were Transformers introduced?",
        answer="Transformers were introduced in 2020.",
        evidence=EVIDENCE,
    )


def test_no_answer_skips_grounding():
    validator = Mock()

    retrieval_result = make_result()
    retrieval_result["agent_output"] = None

    result = validate_generated_answer(
        question="Test question",
        retrieval_result=retrieval_result,
        grounding_validator=validator,
    )

    assert result["grounding"] is None
    validator.validate.assert_not_called()


def test_previous_failure_skips_grounding():
    validator = Mock()

    retrieval_result = make_result()
    retrieval_result["error"] = {
        "type": "relevance_grading_error",
        "message": "Grading failed.",
    }

    result = validate_generated_answer(
        question="Test question",
        retrieval_result=retrieval_result,
        grounding_validator=validator,
    )

    assert result["grounding"] is None
    validator.validate.assert_not_called()


def test_grounding_failure_preserves_answer():
    validator = Mock()
    validator.validate.side_effect = ValueError(
        "Invalid grounding JSON"
    )

    result = validate_generated_answer(
        question="Test question",
        retrieval_result=make_result(),
        grounding_validator=validator,
    )

    assert result["grounding"] is None
    assert result["evidence"] == EVIDENCE
    assert result["agent_output"] == (
        "Transformers were introduced in 2020."
    )
    assert result["error"]["type"] == (
        "grounding_validation_error"
    )


def test_unexpected_grounding_error_propagates():
    validator = Mock()
    validator.validate.side_effect = RuntimeError(
        "Unexpected error"
    )

    try:
        validate_generated_answer(
            question="Test question",
            retrieval_result=make_result(),
            grounding_validator=validator,
        )
    except RuntimeError as exc:
        assert str(exc) == "Unexpected error"
    else:
        raise AssertionError(
            "Unexpected errors should propagate"
        )
