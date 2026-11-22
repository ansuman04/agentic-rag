
from unittest.mock import Mock

import pytest

from src.answer_corrector import AnswerCorrector


EVIDENCE = [
    {
        "text": "Transformers were introduced in 2017.",
        "source": "sample.pdf",
        "page": 1,
    }
]


def grounding(status):
    return {"grounding_status": status, "claims": []}


def test_grounded_answer_skips_correction():
    corrector = Mock()
    validator = Mock()

    service = AnswerCorrector(corrector, validator)

    result = service.correct(
        "When were Transformers introduced?",
        "Transformers were introduced in 2017.",
        EVIDENCE,
        grounding("grounded"),
    )

    assert result["correction_status"] == "not_required"
    assert result["attempts"] == 0
    assert result["verified_answer"] is not None
    corrector.assert_not_called()
    validator.validate.assert_not_called()


def test_unsupported_answer_is_corrected():
    corrector = Mock(
        return_value="Transformers were introduced in 2017."
    )

    validator = Mock()
    validator.validate.return_value = grounding("grounded")

    service = AnswerCorrector(corrector, validator)

    result = service.correct(
        "When were Transformers introduced?",
        "Transformers were introduced in 2020.",
        EVIDENCE,
        grounding("unsupported"),
    )

    assert result["correction_status"] == "corrected"
    assert result["attempts"] == 1
    assert result["verified_answer"] == (
        "Transformers were introduced in 2017."
    )

    corrector.assert_called_once()
    validator.validate.assert_called_once()


def test_still_ungrounded_after_correction():
    corrector = Mock(
        return_value="Transformers were introduced in 2019."
    )

    validator = Mock()
    validator.validate.return_value = grounding("unsupported")

    service = AnswerCorrector(corrector, validator)

    result = service.correct(
        "When were Transformers introduced?",
        "Transformers were introduced in 2020.",
        EVIDENCE,
        grounding("unsupported"),
    )

    assert result["correction_status"] == "still_ungrounded"
    assert result["verified_answer"] is None
    assert result["attempts"] == 1
    corrector.assert_called_once()


def test_partially_grounded_answer_is_corrected():
    corrector = Mock(return_value="Verified answer")

    validator = Mock()
    validator.validate.return_value = grounding("grounded")

    service = AnswerCorrector(corrector, validator)

    result = service.correct(
        "Test question",
        "Partially supported answer",
        EVIDENCE,
        grounding("partially_grounded"),
    )

    assert result["correction_status"] == "corrected"
    assert result["verified_answer"] == "Verified answer"


def test_empty_correction_is_not_verified():
    corrector = Mock(return_value=" ")
    validator = Mock()

    service = AnswerCorrector(corrector, validator)

    result = service.correct(
        "Test question",
        "Unsupported answer",
        EVIDENCE,
        grounding("unsupported"),
    )

    assert result["correction_status"] == "correction_failed"
    assert result["verified_answer"] is None
    validator.validate.assert_not_called()


def test_correction_requires_evidence():
    service = AnswerCorrector(
        corrector=Mock(),
        grounding_validator=Mock(),
    )

    with pytest.raises(ValueError, match="evidence"):
        service.correct(
            "Test question",
            "Test answer",
            [],
            grounding("unsupported"),
        )
