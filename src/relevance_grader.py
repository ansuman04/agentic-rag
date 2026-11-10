
from collections.abc import Callable


class RelevanceGrader:
    """
    Grade retrieved evidence before answer generation.

    The scorer receives one question and all evidence,
    then returns a relevance decision for each result.
    """

    def __init__(
        self,
        scorer: Callable[[str, list[dict]], list[bool]],
    ):
        self.scorer = scorer

    def grade(
        self,
        question: str,
        evidence: list[dict],
    ) -> dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not isinstance(evidence, list):
            raise ValueError("Evidence must be a list.")

        if any(not isinstance(item, dict) for item in evidence):
            raise ValueError(
                "Each evidence item must be a dictionary."
            )

        if not evidence:
            return {
                "relevant_evidence": [],
                "rejected_evidence": [],
            }

        decisions = self.scorer(
            question.strip(),
            evidence,
        )

        if not isinstance(decisions, list):
            raise ValueError(
                "Scorer must return a list of decisions."
            )

        if len(decisions) != len(evidence):
            raise ValueError(
                "Number of grading decisions must "
                "match number of evidence items."
            )

        if any(type(item) is not bool for item in decisions):
            raise ValueError(
                "Each grading decision must be a boolean."
            )

        relevant = []
        rejected = []

        for item, is_relevant in zip(evidence, decisions):
            if is_relevant:
                relevant.append(item)
            else:
                rejected.append(item)

        return {
            "relevant_evidence": relevant,
            "rejected_evidence": rejected,
        }
