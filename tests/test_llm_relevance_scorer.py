
from unittest.mock import patch

import pytest

from src.llm_relevance_scorer import LLMRelevanceScorer


EVIDENCE = [
    {
        "text": "Positional encoding represents positions.",
        "source": "sample.pdf",
        "page": 6,
    },
    {
        "text": "Training used eight GPUs.",
        "source": "sample.pdf",
        "page": 10,
    },
]


@pytest.fixture
def scorer():
    with (
        patch(
            "src.llm_relevance_scorer."
            "create_relevance_grader_agent"
        ),
        patch(
            "src.llm_relevance_scorer."
            "create_relevance_grading_task"
        ),
    ):
        yield LLMRelevanceScorer()


def test_valid_llm_response(scorer):
    scorer.agent_runner = (
        lambda agent, task: "[true, false]"
    )

    assert scorer.score(
        "Why use positional encoding?",
        EVIDENCE,
    ) == [True, False]


def test_invalid_json_is_rejected(scorer):
    scorer.agent_runner = (
        lambda agent, task: "not valid json"
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        scorer.score("Test question", EVIDENCE)


def test_wrong_decision_count_is_rejected(scorer):
    scorer.agent_runner = (
        lambda agent, task: "[true]"
    )

    with pytest.raises(ValueError, match="does not match"):
        scorer.score("Test question", EVIDENCE)


def test_non_boolean_decisions_are_rejected(scorer):
    scorer.agent_runner = (
        lambda agent, task: '["yes", "no"]'
    )

    with pytest.raises(ValueError, match="booleans"):
        scorer.score("Test question", EVIDENCE)


def test_json_object_is_rejected(scorer):
    scorer.agent_runner = (
        lambda agent, task: '{"relevant": true}'
    )

    with pytest.raises(ValueError, match="JSON array"):
        scorer.score("Test question", EVIDENCE)


def test_empty_evidence_skips_llm(scorer):
    def fail_if_called(agent, task):
        raise AssertionError("LLM must not be called")

    scorer.agent_runner = fail_if_called

    assert scorer.score("Test question", []) == []


def test_empty_question_is_rejected(scorer):
    scorer.agent_runner = (
        lambda agent, task: "[true, false]"
    )

    with pytest.raises(ValueError, match="Question"):
        scorer.score(" ", EVIDENCE)


def test_boolean_strings_are_normalized(scorer):
    scorer.agent_runner = (
        lambda agent, task: '["true", "false"]'
    )

    result = scorer.score(
        "Why use positional encoding?",
        EVIDENCE,
    )

    assert result == [True, False]

