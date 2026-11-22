"""Public HTTP contracts; raw generated answers are deliberately excluded."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Annotated[str, StringConstraints(
        strict=True, strip_whitespace=True, min_length=1, max_length=10000,
    )]


class QuestionResponse(BaseModel):
    question: str
    route: Literal["vectorstore", "websearch"]
    status: Literal[
        "verified", "unverified", "no_evidence", "no_relevant_evidence",
        "grading_failed", "generation_failed", "grounding_failed", "correction_failed",
    ]
    answer: str | None
    # Preserve the existing document/web evidence shapes without losing metadata.
    sources: list[dict[str, Any]]
    error: dict[str, Any] | None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
