
from unittest.mock import patch

import pytest

from src.llm_grounding_validator import LLMGroundingScorer


EVIDENCE = [
    {
        "text": "Transformers use positional encoding.",
        "source": "sample.pdf",
        "page": 6,
    }
]


@pytest.fixture
def scorer_factory():
    with (
        patch(
            "src.llm_grounding_validator."
            "create_grounding_validator_agent"
        ),
        patch(
            "src.llm_grounding_validator."
            "create_grounding_validation_task"
        ),
    ):
        def create(raw_output):
            return LLMGroundingScorer(
                agent_runner=lambda agent, task: raw_output
            )

        yield create


def test_valid_grounding_response(scorer_factory):
    response = """
    [
        {
            "claim": "Transformers use positional encoding.",
            "verdict": "supported",
            "reason": "Explicitly supported by the evidence."
        }
    ]
    """

    scorer = scorer_factory(response)

    result = scorer.score(
        "Why use positional encoding?",
        "Transformers use positional encoding.",
        EVIDENCE,
    )

    assert len(result) == 1
    assert result[0]["verdict"] == "supported"


def test_multiple_claims(scorer_factory):
    response = """
    [
        {
            "claim": "Transformers use positional encoding.",
            "verdict": "supported",
            "reason": "The passage confirms this."
        },
        {
            "claim": "Transformers were introduced in 2020.",
            "verdict": "insufficient_evidence",
            "reason": "The passage provides no release year."
        }
    ]
    """

    scorer = scorer_factory(response)

    result = scorer.score(
        "Explain Transformers",
        "Transformers use positional encoding.",
        EVIDENCE,
    )

    assert len(result) == 2
    assert result[1]["verdict"] == "insufficient_evidence"


def test_invalid_json_is_rejected(scorer_factory):
    scorer = scorer_factory("Invalid JSON")

    with pytest.raises(ValueError, match="Invalid grounding JSON"):
        scorer.score("Question", "Answer", EVIDENCE)


def test_json_object_is_rejected(scorer_factory):
    scorer = scorer_factory('{"verdict": "supported"}')

    with pytest.raises(ValueError, match="non-empty JSON array"):
        scorer.score("Question", "Answer", EVIDENCE)


def test_invalid_verdict_is_rejected(scorer_factory):
    response = """
    [
        {
            "claim": "Example claim",
            "verdict": "maybe",
            "reason": "Uncertain"
        }
    ]
    """

    scorer = scorer_factory(response)

    with pytest.raises(ValueError, match="invalid verdict"):
        scorer.score("Question", "Answer", EVIDENCE)


def test_missing_fields_are_rejected(scorer_factory):
    response = """
    [
        {
            "claim": "Example claim",
            "verdict": "supported"
        }
    ]
    """

    scorer = scorer_factory(response)

    with pytest.raises(ValueError, match="invalid fields"):
        scorer.score("Question", "Answer", EVIDENCE)


def test_empty_evidence_skips_llm():
    runner_called = False

    def runner(agent, task):
        nonlocal runner_called
        runner_called = True
        return "[]"

    with patch(
        "src.llm_grounding_validator."
        "create_grounding_validator_agent"
    ):
        scorer = LLMGroundingScorer(
            agent_runner=runner,
        )

    result = scorer.score(
        "Question",
        "Answer",
        [],
    )

    assert result == []
    assert runner_called is False


def test_empty_answer_is_rejected(scorer_factory):
    scorer = scorer_factory("[]")

    with pytest.raises(ValueError, match="Answer"):
        scorer.score("Question", " ", EVIDENCE)
