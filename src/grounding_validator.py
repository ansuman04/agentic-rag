
from collections.abc import Callable


VALID_VERDICTS = {
    "supported",
    "unsupported",
    "insufficient_evidence",
}


class GroundingValidator:
    """
    Validate whether an answer's factual claims
    are supported by retrieved evidence.

    The validator function will later be backed
    by a CrewAI agent using Qwen2.5.
    """

    def __init__(
        self,
        validator: Callable[
            [str, str, list[dict]],
            list[dict],
        ],
    ):
        self.validator = validator

    def validate(
        self,
        question: str,
        answer: str,
        evidence: list[dict],
    ) -> dict:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not isinstance(evidence, list):
            raise ValueError("Evidence must be a list.")

        if any(
            not isinstance(item, dict)
            for item in evidence
        ):
            raise ValueError(
                "Each evidence item must be a dictionary."
            )

        if not evidence:
            return {
                "grounding_status": "no_evidence",
                "claims": [],
                "supported_claims": [],
                "unsupported_claims": [],
                "insufficient_claims": [],
            }

        claims = self.validator(
            question.strip(),
            answer.strip(),
            evidence,
        )

        if not isinstance(claims, list) or not claims:
            raise ValueError(
                "Validator must return a non-empty "
                "list of claim assessments."
            )

        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError(
                    "Each claim assessment must be a dictionary."
                )

            if (
                not isinstance(claim.get("claim"), str)
                or not claim["claim"].strip()
            ):
                raise ValueError(
                    "Each assessment must contain a claim."
                )

            if claim.get("verdict") not in VALID_VERDICTS:
                raise ValueError(
                    "Invalid grounding verdict."
                )

            if not isinstance(
                claim.get("reason"), str
            ) or not claim["reason"].strip():
                raise ValueError(
                    "Each assessment must contain a reason."
                )

        supported = [
            item for item in claims
            if item["verdict"] == "supported"
        ]

        unsupported = [
            item for item in claims
            if item["verdict"] == "unsupported"
        ]

        insufficient = [
            item for item in claims
            if item["verdict"] == "insufficient_evidence"
        ]

        if unsupported:
            status = "unsupported"

        elif insufficient:
            status = "partially_grounded"

        else:
            status = "grounded"

        return {
            "grounding_status": status,
            "claims": claims,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "insufficient_claims": insufficient,
        }
