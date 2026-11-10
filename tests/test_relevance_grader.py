
import pytest

from src.relevance_grader import RelevanceGrader


def test_separates_relevant_and_irrelevant_evidence():
    evidence = [
        {
            "text": "Positional encoding represents token positions.",
            "source": "sample.pdf",
            "page": 6,
        },
        {
            "text": "The model uses eight attention heads.",
            "source": "sample.pdf",
            "page": 4,
        },
    ]

    calls = []

    def scorer(question, items):
        calls.append((question, items))
        return [True, False]

    grader = RelevanceGrader(scorer=scorer)

    result = grader.grade(
        "Why use positional encoding?",
        evidence,
    )

    assert result["relevant_evidence"] == [evidence[0]]
    assert result["rejected_evidence"] == [evidence[1]]
    assert len(calls) == 1


def test_web_evidence_is_supported():
    evidence = [
        {
            "title": "Bengaluru Weather",
            "url": "https://example.com/weather",
            "snippet": "The temperature is 23 degrees.",
        }
    ]

    grader = RelevanceGrader(
        scorer=lambda question, items: [True]
    )

    result = grader.grade(
        "What is Bengaluru's temperature?",
        evidence,
    )

    assert result["relevant_evidence"] == evidence
    assert result["rejected_evidence"] == []


def test_empty_evidence_skips_scorer():
    def scorer(question, evidence):
        raise AssertionError("Scorer should not be called")

    grader = RelevanceGrader(scorer)

    result = grader.grade("Test question", [])

    assert result["relevant_evidence"] == []
    assert result["rejected_evidence"] == []


def test_rejects_empty_question():
    grader = RelevanceGrader(
        scorer=lambda question, evidence: [True]
    )

    with pytest.raises(ValueError):
        grader.grade("   ", [{"text": "Sample"}])


def test_rejects_incorrect_number_of_decisions():
    grader = RelevanceGrader(
        scorer=lambda question, evidence: [True]
    )

    with pytest.raises(ValueError):
        grader.grade(
            "Test question",
            [{"text": "A"}, {"text": "B"}],
        )


def test_rejects_invalid_decision_type():
    grader = RelevanceGrader(
        scorer=lambda question, evidence: ["yes"]
    )

    with pytest.raises(ValueError):
        grader.grade("Test question", [{"text": "A"}])
