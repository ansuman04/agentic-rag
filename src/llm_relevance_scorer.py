
import json
from collections.abc import Callable

from src.agents import create_relevance_grader_agent
from src.tasks import create_relevance_grading_task


class LLMRelevanceScorer:
    """
    Uses CrewAI and a local LLM to classify retrieved
    evidence as relevant or irrelevant.
    """

    def __init__(
        self,
        agent_runner: Callable | None = None,
    ):
        self.agent = create_relevance_grader_agent()
        self.agent_runner = agent_runner

    def score(
        self,
        question: str,
        evidence: list[dict],
    ) -> list[bool]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not evidence:
            return []

        task = create_relevance_grading_task(
            agent=self.agent,
            question=question,
            evidence=evidence,
        )

        if self.agent_runner is not None:
            raw_output = self.agent_runner(
                self.agent,
                task,
            )
        else:
            # Import here to avoid circular imports.
            from src.orchestrator import run_agent

            raw_output = run_agent(
                self.agent,
                task,
            )

        return self._parse_decisions(
            raw_output,
            expected_count=len(evidence),
        )

    
    @staticmethod
    def _parse_decisions(
        raw_output: str,
        expected_count: int,
    ) -> list[bool]:
        if not isinstance(raw_output, str):
            raise ValueError(
                "LLM grading response must be a string."
            )

        try:
            decisions = json.loads(raw_output.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON for relevance grading."
            ) from exc

        if not isinstance(decisions, list):
            raise ValueError(
                "LLM grading response must be a JSON array."
            )

        if len(decisions) != expected_count:
            raise ValueError(
                "Number of LLM decisions does not match "
                "the number of evidence items."
            )

        normalized = []

        for decision in decisions:
            if type(decision) is bool:
                normalized.append(decision)

            elif isinstance(decision, str):
                value = decision.strip().lower()

                if value == "true":
                    normalized.append(True)
                elif value == "false":
                    normalized.append(False)
                else:
                    raise ValueError(
                        "All LLM grading decisions must "
                        "be booleans or boolean strings."
                    )

            else:
                raise ValueError(
                    "All LLM grading decisions must "
                    "be booleans or boolean strings."
                )

        return normalized
