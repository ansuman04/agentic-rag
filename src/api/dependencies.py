"""Lazy, thread-safe ownership of the existing local orchestrator."""

from threading import RLock
from typing import TYPE_CHECKING

from fastapi import Request

from .schemas import QuestionRequest

if TYPE_CHECKING:
    from src.orchestrator import AgentOrchestrator


def build_local_orchestrator() -> "AgentOrchestrator":
    # Keep health checks and API imports independent of model initialization.
    from src.orchestrator import build_local_orchestrator as build
    return build()


class OrchestratorProvider:
    def __init__(self):
        self.lock = RLock()
        self._orchestrator: "AgentOrchestrator | None" = None

    def get(self) -> "AgentOrchestrator":
        with self.lock:
            if self._orchestrator is None:
                self._orchestrator = build_local_orchestrator()
            return self._orchestrator


def get_orchestrator(request: Request, body: QuestionRequest) -> "AgentOrchestrator":
    # Requiring the same body here validates it before expensive initialization.
    return request.app.state.orchestrator_provider.get()
