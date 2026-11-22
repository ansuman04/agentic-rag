
from unittest.mock import Mock, patch

import pytest

from src.llm_answer_corrector import LLMAnswerCorrector


@pytest.fixture(autouse=True)
def mock_answer_correction_task():
    """
    Prevent the real CrewAI Task from being created
    during LLMAnswerCorrector unit tests.
    """
    with patch(
        "src.llm_answer_corrector."
        "create_answer_correction_task"
    ) as mock_task:
        yield mock_task


@pytest.fixture
def service_factory():
    with (
        patch(
            "src.llm_answer_corrector."
            "create_answer_correction_agent"
        ),
        patch(
            "src.llm_answer_corrector."
            "create_answer_correction_task"
        ),
    ):
        def create_service(runner):
            return LLMAnswerCorrector(
                agent_runner=runner,
            )

        yield create_service


EVIDENCE = [
    {
        "text": "Transformers were introduced in 2017.",
        "source": "sample.pdf",
        "page": 1,
    }
]

GROUNDING = {
    "grounding_status": "unsupported",
    "claims": [
        {
            "claim": "Transformers were introduced in 2020.",
            "verdict": "unsupported",
            "reason": "Evidence states 2017.",
        }
    ],
}

def create_service(runner):
    agent_patcher = patch(
        "src.llm_answer_corrector."
        "create_answer_correction_agent"
    )

    task_patcher = patch(
        "src.llm_answer_corrector."
        "create_answer_correction_task"
    )

    agent_patcher.start()
    task_patcher.start()

    try:
        service = LLMAnswerCorrector(
            agent_runner=runner,
        )
    finally:
        task_patcher.stop()
        agent_patcher.stop()

    return service


def test_correct_answer_successfully(service_factory):
    runner = Mock(
        return_value="Transformers were introduced in 2017."
    )

    service = service_factory(runner)


def test_correct_answer_trims_whitespace():
    runner = Mock(
        return_value="  Corrected answer.  "
    )
    service = create_service(runner)

    result = service.correct(
        "Test question",
        "Original answer",
        EVIDENCE,
        GROUNDING,
    )

    assert result == "Corrected answer."


def test_empty_question_rejected():
    service = create_service(Mock())

    with pytest.raises(ValueError, match="Question"):
        service.correct(
            " ",
            "Original answer",
            EVIDENCE,
            GROUNDING,
        )


def test_empty_answer_rejected():
    service = create_service(Mock())

    with pytest.raises(ValueError, match="Answer"):
        service.correct(
            "Test question",
            " ",
            EVIDENCE,
            GROUNDING,
        )


def test_empty_evidence_rejected():
    service = create_service(Mock())

    with pytest.raises(ValueError, match="evidence"):
        service.correct(
            "Test question",
            "Original answer",
            [],
            GROUNDING,
        )


def test_invalid_grounding_rejected():
    service = create_service(Mock())

    with pytest.raises(ValueError, match="Grounding"):
        service.correct(
            "Test question",
            "Original answer",
            EVIDENCE,
            None,
        )


def test_empty_llm_response_rejected():
    service = create_service(
        Mock(return_value="   ")
    )

    with pytest.raises(
        ValueError,
        match="Correction response cannot be empty",
    ):
        service.correct(
            "Test question",
            "Original answer",
            EVIDENCE,
            GROUNDING,
        )


def test_non_string_llm_response_rejected():
    service = create_service(
        Mock(return_value={"answer": "incorrect format"})
    )

    with pytest.raises(
        ValueError,
        match="Correction response must be a string",
    ):
        service.correct(
            "Test question",
            "Original answer",
            EVIDENCE,
            GROUNDING,
        )


def test_task_receives_original_grounding():
    runner = Mock(return_value="Corrected answer")

    service = create_service(runner)

    with patch(
        "src.llm_answer_corrector."
        "create_answer_correction_task"
    ) as mock_task:
        service.correct(
            "Test question",
            "Original answer",
            EVIDENCE,
            GROUNDING,
        )

    assert mock_task.call_args.kwargs["grounding"] == GROUNDING
    assert mock_task.call_args.kwargs["evidence"] == EVIDENCE
