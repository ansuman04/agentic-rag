
from unittest.mock import patch

from src.agents import create_answer_correction_agent
from src.tasks import create_answer_correction_task


EVIDENCE = [
    {
        "text": (
            "The Transformer architecture was "
            "introduced in 2017."
        ),
        "source": "sample.pdf",
        "page": 1,
    }
]

GROUNDING = {
    "grounding_status": "unsupported",
    "claims": [
        {
            "claim": (
                "The Transformer architecture was "
                "introduced in 2020."
            ),
            "verdict": "unsupported",
            "reason": (
                "Evidence establishes 2017, not 2020."
            ),
        }
    ],
}


def test_answer_correction_agent_configuration():
    with (
        patch("src.agents.LLM") as mock_llm,
        patch("src.agents.Agent") as mock_agent,
    ):
        create_answer_correction_agent()

        mock_llm.assert_called_once()
        mock_agent.assert_called_once()

        kwargs = mock_agent.call_args.kwargs

        assert kwargs["role"] == (
            "Answer Correction Specialist"
        )
        assert kwargs["tools"] == []
        assert kwargs["allow_delegation"] is False
        assert kwargs["llm"] is mock_llm.return_value


def test_correction_task_contains_required_context():
    with patch("src.tasks.Task") as mock_task:
        create_answer_correction_task(
            agent=object(),
            question=(
                "When were Transformers introduced?"
            ),
            answer=(
                "Transformers were introduced in 2020."
            ),
            evidence=EVIDENCE,
            grounding=GROUNDING,
        )

        kwargs = mock_task.call_args.kwargs
        description = kwargs["description"]

        assert (
            "When were Transformers introduced?"
            in description
        )
        assert (
            "Transformers were introduced in 2020."
            in description
        )
        assert "sample.pdf" in description
        assert "2017" in description
        assert "unsupported" in description


def test_correction_task_prevents_new_retrieval():
    with patch("src.tasks.Task") as mock_task:
        create_answer_correction_task(
            agent=object(),
            question="Test question",
            answer="Test answer",
            evidence=EVIDENCE,
            grounding=GROUNDING,
        )

        description = (
            mock_task.call_args.kwargs["description"]
        )

        assert (
            "Do not perform additional retrieval"
            in description
        )
        assert "Do not introduce new facts" in description
        assert "Do not rely on your own knowledge" in description
        assert "Return only the corrected answer" in description
