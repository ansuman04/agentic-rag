
from unittest.mock import Mock

from src.orchestrator import (
    apply_answer_correction,
    build_retrieval_response,
)


QUESTION = "When were Transformers introduced?"
ORIGINAL = "Transformers were introduced in 2020."
CORRECTED = "Transformers were introduced in 2017."

EVIDENCE = [
    {
        "text": "Transformers were introduced in 2017.",
        "source": "sample.pdf",
        "page": 1,
    }
]


def grounding(status):
    return {
        "grounding_status": status,
        "claims": [],
    }


def retrieval_result(status="unsupported"):
    return {
        "evidence": EVIDENCE,
        "relevant_evidence": EVIDENCE,
        "rejected_evidence": [],
        "agent_output": ORIGINAL,
        "grounding": grounding(status),
        "error": None,
    }


def test_successful_correction_produces_verified_answer():
    corrector = Mock()
    corrector.correct.return_value = {
        "original_answer": ORIGINAL,
        "corrected_answer": CORRECTED,
        "verified_answer": CORRECTED,
        "correction_status": "corrected",
        "attempts": 1,
        "grounding": grounding("grounded"),
    }

    result = apply_answer_correction(
        QUESTION,
        retrieval_result(),
        corrector,
    )

    assert result["original_answer"] == ORIGINAL
    assert result["agent_output"] == ORIGINAL
    assert result["verified_answer"] == CORRECTED
    assert result["grounding"]["grounding_status"] == "grounded"

    response = build_retrieval_response(
        QUESTION,
        "vectorstore",
        result,
    )

    assert response["status"] == "verified"
    corrector.correct.assert_called_once()


def test_failed_correction_is_not_verified():
    corrector = Mock()
    corrector.correct.return_value = {
        "original_answer": ORIGINAL,
        "corrected_answer": "Introduced in 2019.",
        "verified_answer": None,
        "correction_status": "still_ungrounded",
        "attempts": 1,
        "grounding": grounding("unsupported"),
    }

    result = apply_answer_correction(
        QUESTION,
        retrieval_result(),
        corrector,
    )

    assert result["verified_answer"] is None
    assert result["agent_output"] == ORIGINAL

    response = build_retrieval_response(
        QUESTION,
        "websearch",
        result,
    )

    assert response["status"] == "unverified"


def test_grounded_answer_needs_no_llm_retry():
    from src.answer_corrector import AnswerCorrector

    llm_corrector = Mock()
    validator = Mock()

    service = AnswerCorrector(
        corrector=llm_corrector,
        grounding_validator=validator,
    )

    result = apply_answer_correction(
        QUESTION,
        retrieval_result("grounded"),
        service,
    )

    assert result["verified_answer"] == ORIGINAL
    assert result["correction"]["attempts"] == 0

    llm_corrector.assert_not_called()
    validator.validate.assert_not_called()


def test_upstream_error_skips_correction():
    corrector = Mock()
    data = retrieval_result()
    data["error"] = {
        "type": "relevance_grading_error",
        "message": "Grading failed.",
    }

    result = apply_answer_correction(
        QUESTION,
        data,
        corrector,
    )

    assert result["verified_answer"] is None
    assert result["correction"] is None
    corrector.correct.assert_not_called()


def test_correction_error_is_recorded():
    corrector = Mock()
    corrector.correct.side_effect = ValueError(
        "Invalid correction response"
    )

    result = apply_answer_correction(
        QUESTION,
        retrieval_result(),
        corrector,
    )

    assert result["error"]["type"] == "answer_correction_error"
    assert result["verified_answer"] is None

    response = build_retrieval_response(
        QUESTION,
        "vectorstore",
        result,
    )

    assert response["status"] == "correction_failed"


def test_no_grounding_skips_correction():
    corrector = Mock()
    data = retrieval_result()
    data["grounding"] = None

    result = apply_answer_correction(
        QUESTION,
        data,
        corrector,
    )

    assert result["verified_answer"] is None
    corrector.correct.assert_not_called()


def test_unexpected_error_propagates():
    corrector = Mock()
    corrector.correct.side_effect = RuntimeError(
        "Unexpected programming error"
    )

    try:
        apply_answer_correction(
            QUESTION,
            retrieval_result(),
            corrector,
        )
    except RuntimeError as exc:
        assert str(exc) == "Unexpected programming error"
    else:
        raise AssertionError(
            "Unexpected errors should propagate"
        )
