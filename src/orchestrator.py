
from collections.abc import Callable

from crewai import Crew, Process

from src.agents import (
    create_router_agent,
    create_retriever_agent,
)
from src.tasks import (
    create_routing_task,
    create_retrieval_task,
)
from src.tools import DocumentSearchTool
from src.vector_store import VectorStore


class AgentOrchestrator:
    """Route user questions to the appropriate retrieval path."""

    def __init__(
        self,
        router: Callable[[str], str],
        retriever: Callable[[str], str],
    ):
        self.router = router
        self.retriever = retriever

    def run(self, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()
        route = self.router(question).strip().lower()

        if route == "vectorstore":
            evidence = self.retriever(question)
            return {
                "route": route,
                "question": question,
                "evidence": evidence,
            }

        if route == "websearch":
            return {
                "route": route,
                "question": question,
                "evidence": None,
                "message": (
                    "Web search is not implemented yet."
                ),
            }

        raise ValueError(
            f"Unsupported routing decision: {route!r}"
        )


def run_agent(agent, task) -> str:
    """Execute a single CrewAI agent task."""
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )

    result = crew.kickoff()
    return result.raw


def build_local_orchestrator() -> AgentOrchestrator:
    """Connect existing CrewAI agents to local ChromaDB."""

    store = VectorStore()

    if store.collection.count() == 0:
        raise RuntimeError(
            "No indexed documents found. Run python main.py first."
        )

    search_tool = DocumentSearchTool(vector_store=store)

    router_agent = create_router_agent()
    retriever_agent = create_retriever_agent(search_tool)

    def route_question(question: str) -> str:
        task = create_routing_task(router_agent, question)
        return run_agent(router_agent, task)

    def retrieve_question(question: str) -> str:
        task = create_retrieval_task(
            retriever_agent,
            question,
        )
        return run_agent(retriever_agent, task)

    return AgentOrchestrator(
        router=route_question,
        retriever=retrieve_question,
    )