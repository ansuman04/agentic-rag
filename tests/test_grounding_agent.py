
from unittest.mock import patch

from src.agents import create_grounding_validator_agent
from src.tasks import create_grounding_validation_task


def test_grounding_agent_configuration():
    with (
        patch("src.agents.LLM") as mock_llm,
        patch("src.agents.Agent") as mock_agent,
    ):
        create_grounding_validator_agent()

        mock_llm.assert_called_once()
        mock_agent.assert_called_once()

        kwargs = mock_agent.call_args.kwargs

        assert kwargs["role"] == "Answer Grounding Validator"
        assert kwargs["tools"] == []
        assert kwargs["allow_delegation"] is False
        assert kwargs["llm"] is mock_llm.return_value


def test_grounding_task_contains_answer_and_evidence():
    evidence = [
        {
            "text": "Transformers use positional encoding.",
            "source": "sample.pdf",
            "page": 6,
        }
    ]

    with patch("src.tasks.Task") as mock_task:
        create_grounding_validation_task(
            agent=object(),
            question="Why use positional encoding?",
            answer="Transformers use positional encoding.",
            evidence=evidence,
        )

        mock_task.assert_called_once()

        kwargs = mock_task.call_args.kwargs
        description = kwargs["description"]

        assert "Why use positional encoding?" in description
        assert "Transformers use positional encoding." in description
        assert "sample.pdf" in description
        assert '"page": 6' in description
        assert "supported" in description
        assert "unsupported" in description
        assert "insufficient_evidence" in description


def test_grounding_task_requires_json_output():
    with patch("src.tasks.Task") as mock_task:
        create_grounding_validation_task(
            agent=object(),
            question="Test question",
            answer="Test answer",
            evidence=[{"text": "Test evidence"}],
        )

        kwargs = mock_task.call_args.kwargs
        description = kwargs["description"]

        assert "valid JSON array" in description
        assert '"claim"' in description
        assert '"verdict"' in description
        assert '"reason"' in description
        assert "Do not rewrite the answer" in description
