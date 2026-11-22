"""Lazy orchestration resources; admission happens before initialization."""

from threading import Lock, BoundedSemaphore
from typing import TYPE_CHECKING

from fastapi import Request

from src.settings import Settings
from .schemas import QuestionRequest

if TYPE_CHECKING:
    from src.orchestrator import AgentOrchestrator


def build_local_orchestrator() -> "AgentOrchestrator":
    from src.orchestrator import build_local_orchestrator as build
    return build()


class OrchestratorProvider:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.slots = BoundedSemaphore(self.settings.max_active)
        self.lock = Lock()
        self._orchestrator: "AgentOrchestrator | None" = None
        self._pool = []

    def get(self) -> "AgentOrchestrator":
        # Retained for direct callers and P0.1 dependency tests.
        with self.lock:
            if self._orchestrator is None:
                self._orchestrator = build_local_orchestrator()
            return self._orchestrator

    def checkout(self):
        with self.lock:
            if self._pool:
                return self._pool.pop()
        # Each admitted concurrent execution owns separate mutable CrewAI agents.
        return build_local_orchestrator()

    def checkin(self, orchestrator):
        with self.lock:
            self._pool.append(orchestrator)


async def get_orchestrator(request: Request, body: QuestionRequest):
    # Resolve lazily inside the admitted worker, after request validation.
    return request.app.state.orchestrator_provider
