
from collections.abc import Callable

from src.grounding_validator import GroundingValidator


class AnswerCorrector:
    """
    Attempt to correct an ungrounded answer once,
    then verify the correction against the evidence.
    """

    def __init__(
        self,
        corrector: Callable[[str, str, list[dict], dict], str],
        grounding_validator: GroundingValidator,
    ):
        self.corrector = corrector
        self.grounding_validator = grounding_validator

    def correct(
        self,
        question: str,
        answer: str,
        evidence: list[dict],
        initial_grounding: dict,
    ) -> dict:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not isinstance(evidence, list):
            raise ValueError("Evidence must be a list.")

        if not evidence:
            raise ValueError(
                "Correction requires relevant evidence."
            )

        status = initial_grounding.get("grounding_status")

        if status not in {
            "grounded",
            "unsupported",
            "partially_grounded",
        }:
            raise ValueError("Invalid initial grounding status.")

        # Already verified: no correction necessary.
        if status == "grounded":
            return {
                "original_answer": answer,
                "corrected_answer": None,
                "verified_answer": answer,
                "correction_status": "not_required",
                "attempts": 0,
                "grounding": initial_grounding,
            }

        # One correction attempt only.
        corrected_answer = self.corrector(
            question,
            answer,
            evidence,
            initial_grounding,
        )

        if (
            not isinstance(corrected_answer, str)
            or not corrected_answer.strip()
        ):
            return {
                "original_answer": answer,
                "corrected_answer": None,
                "verified_answer": None,
                "correction_status": "correction_failed",
                "attempts": 1,
                "grounding": initial_grounding,
            }

        corrected_answer = corrected_answer.strip()

        # Validate corrected answer using the same evidence.
        new_grounding = self.grounding_validator.validate(
            question=question,
            answer=corrected_answer,
            evidence=evidence,
        )

        if new_grounding["grounding_status"] == "grounded":
            correction_status = "corrected"
            verified_answer = corrected_answer
        else:
            correction_status = "still_ungrounded"
            verified_answer = None

        return {
            "original_answer": answer,
            "corrected_answer": corrected_answer,
            "verified_answer": verified_answer,
            "correction_status": correction_status,
            "attempts": 1,
            "grounding": new_grounding,
        }
