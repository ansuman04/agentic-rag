
from collections.abc import Callable

from src.agents import create_answer_correction_agent
from src.tasks import create_answer_correction_task


class LLMAnswerCorrector:
    """
    Use a CrewAI agent to correct unsupported
    answers using retrieved evidence.
    """

    def __init__(
        self,
        agent_runner: Callable | None = None,
    ):
        self.agent = create_answer_correction_agent()
        self.agent_runner = agent_runner

    def correct(
        self,
        question: str,
        answer: str,
        evidence: list[dict],
        grounding: dict,
    ) -> str:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not isinstance(evidence, list) or not evidence:
            raise ValueError(
                "Correction requires relevant evidence."
            )

        if any(not isinstance(item, dict) for item in evidence):
            raise ValueError(
                "Each evidence item must be a dictionary."
            )

        if not isinstance(grounding, dict):
            raise ValueError(
                "Grounding result must be a dictionary."
            )

        task = create_answer_correction_task(
            agent=self.agent,
            question=question.strip(),
            answer=answer.strip(),
            evidence=evidence,
            grounding=grounding,
        )

        if self.agent_runner is not None:
            corrected_answer = self.agent_runner(
                self.agent,
                task,
            )
        else:
            # Import here to avoid circular imports.
            from src.orchestrator import run_agent

            corrected_answer = run_agent(
                self.agent,
                task,
            )

        if not isinstance(corrected_answer, str):
            raise ValueError(
                "Correction response must be a string."
            )

        corrected_answer = corrected_answer.strip()

        if not corrected_answer:
            raise ValueError(
                "Correction response cannot be empty."
            )

        return corrected_answer
