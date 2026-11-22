
import pytest

from src.grounding_validator import GroundingValidator


EVIDENCE = [
    {
        "text": (
            "Transformers use sinusoidal positional "
            "encodings to represent token positions."
        ),
        "source": "sample.pdf",
        "page": 6,
    }
]


def test_fully_grounded_answer():
    def validator(question, answer, evidence):
        return [
            {
                "claim": (
                    "Transformers use sinusoidal "
                    "positional encodings."
                ),
                "verdict": "supported",
                "reason": (
                    "The document explicitly states this."
                ),
            }
        ]

    service = GroundingValidator(validator)

    result = service.validate(
        question="Why use positional encoding?",
        answer=(
            "Transformers use sinusoidal "
            "positional encodings."
        ),
        evidence=EVIDENCE,
    )

    assert result["grounding_status"] == "grounded"
    assert len(result["supported_claims"]) == 1
    assert result["unsupported_claims"] == []


def test_unsupported_claim_is_detected():
    def validator(question, answer, evidence):
        return [
            {
                "claim": (
                    "Transformers were introduced in 2020."
                ),
                "verdict": "unsupported",
                "reason": (
                    "The supplied evidence contradicts "
                    "this claim."
                ),
            }
        ]

    service = GroundingValidator(validator)

    result = service.validate(
        "When were Transformers introduced?",
        "Transformers were introduced in 2020.",
        EVIDENCE,
    )

    assert result["grounding_status"] == "unsupported"
    assert len(result["unsupported_claims"]) == 1


def test_insufficient_evidence():
    def validator(question, answer, evidence):
        return [
            {
                "claim": (
                    "The paper received an award."
                ),
                "verdict": "insufficient_evidence",
                "reason": (
                    "The provided passage does not "
                    "mention awards."
                ),
            }
        ]

    service = GroundingValidator(validator)

    result = service.validate(
        "Did the paper receive an award?",
        "The paper received an award.",
        EVIDENCE,
    )

    assert (
        result["grounding_status"]
        == "partially_grounded"
    )
    assert len(result["insufficient_claims"]) == 1


def test_no_evidence_skips_validator():
    def validator(question, answer, evidence):
        raise AssertionError(
            "Validator must not run without evidence."
        )

    service = GroundingValidator(validator)

    result = service.validate(
        "Test question",
        "Test answer",
        [],
    )

    assert result["grounding_status"] == "no_evidence"


def test_empty_answer_is_rejected():
    service = GroundingValidator(
        lambda question, answer, evidence: []
    )

    with pytest.raises(ValueError, match="Answer"):
        service.validate(
            "Test question",
            "   ",
            EVIDENCE,
        )


def test_invalid_verdict_is_rejected():
    def validator(question, answer, evidence):
        return [
            {
                "claim": "Example claim",
                "verdict": "maybe",
                "reason": "Unknown",
            }
        ]

    service = GroundingValidator(validator)

    with pytest.raises(
        ValueError,
        match="Invalid grounding verdict",
    ):
        service.validate(
            "Test question",
            "Test answer",
            EVIDENCE,
        )


def test_empty_claim_assessments_are_rejected():
    service = GroundingValidator(
        lambda question, answer, evidence: []
    )

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        service.validate(
            "Test question",
            "Test answer",
            EVIDENCE,
        )
