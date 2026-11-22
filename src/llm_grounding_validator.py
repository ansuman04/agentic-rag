
import json
from collections.abc import Callable

from src.agents import create_grounding_validator_agent
from src.tasks import create_grounding_validation_task


class LLMGroundingScorer:
    """
    Use a CrewAI agent to assess factual claims
    in an answer against retrieved evidence.
    """

    def __init__(
        self,
        agent_runner: Callable | None = None,
    ):
        self.agent = create_grounding_validator_agent()
        self.agent_runner = agent_runner

    def score(
        self,
        question: str,
        answer: str,
        evidence: list[dict],
    ) -> list[dict]:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not isinstance(evidence, list):
            raise ValueError("Evidence must be a list.")

        if any(not isinstance(item, dict) for item in evidence):
            raise ValueError(
                "Each evidence item must be a dictionary."
            )

        if not evidence:
            return []

        task = create_grounding_validation_task(
            agent=self.agent,
            question=question.strip(),
            answer=answer.strip(),
            evidence=evidence,
        )

        if self.agent_runner is not None:
            raw_output = self.agent_runner(
                self.agent,
                task,
            )
        else:
            # Local import prevents circular dependencies.
            from src.orchestrator import run_agent

            raw_output = run_agent(
                self.agent,
                task,
            )

        return self._parse_response(raw_output)

    @staticmethod
    def _parse_response(raw_output: str) -> list[dict]:
        if not isinstance(raw_output, str):
            raise ValueError(
                "Grounding response must be a string."
            )

        raw_output = raw_output.strip()

        if not raw_output:
            raise ValueError(
                "Grounding response cannot be empty."
            )

        try:
            claims = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid grounding JSON response."
            ) from exc

        if not isinstance(claims, list) or not claims:
            raise ValueError(
                "Grounding response must be a "
                "non-empty JSON array."
            )

        valid_verdicts = {
            "supported",
            "unsupported",
            "insufficient_evidence",
        }

        for index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                raise ValueError(
                    f"Claim {index} must be a JSON object."
                )

            if set(claim) != {
                "claim",
                "verdict",
                "reason",
            }:
                raise ValueError(
                    f"Claim {index} has invalid fields."
                )

            if (
                not isinstance(claim["claim"], str)
                or not claim["claim"].strip()
            ):
                raise ValueError(
                    f"Claim {index} must contain "
                    "non-empty claim text."
                )

            if claim["verdict"] not in valid_verdicts:
                raise ValueError(
                    f"Claim {index} has an invalid verdict."
                )

            if (
                not isinstance(claim["reason"], str)
                or not claim["reason"].strip()
            ):
                raise ValueError(
                    f"Claim {index} must contain a reason."
                )

        return claims
