
from collections.abc import Callable
from src.agents import create_web_search_agent
from src.tasks import create_web_search_task
from src.web_tools import WebSearchTool
from crewai import Crew, Process
from src.web_retrieval import WebRetrievalService
from src.web_tools import WebSearchTool
from src.agents import create_web_summary_agent
from src.tasks import create_web_summary_task

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
    """Route questions to document retrieval or web search."""

    def __init__(
        self,
        router: Callable[[str], str],
        retriever: Callable[[str], str],
        web_search: Callable[[str], str] | None = None,
    ):
        self.router = router
        self.retriever = retriever
        self.web_search = web_search

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
            if self.web_search is None:
                return {
                    "route": route,
                    "question": question,
                    "evidence": None,
                    "message": "Web search is not implemented yet.",
                }

            evidence = self.web_search(question)

            return {
                "route": route,
                "question": question,
                "evidence": evidence,
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
    """Build the local Agentic RAG workflow."""

    store = VectorStore()

    search_tool = DocumentSearchTool(vector_store=store)
    web_search_tool = WebSearchTool()

    router_agent = create_router_agent()
    retriever_agent = create_retriever_agent(search_tool)
    web_retrieval_service = WebRetrievalService(
        tool=web_search_tool
    )
    # LLM-based evidence summarization.
    web_summary_agent = create_web_summary_agent()

    def route_question(question: str) -> str:
        task = create_routing_task(
            router_agent,
            question,
        )
        return run_agent(router_agent, task)

    def retrieve_question(question: str) -> str:
        if store.collection.count() == 0:
            raise RuntimeError(
                "No indexed documents found. Run python main.py first."
            )

        task = create_retrieval_task(
            retriever_agent,
            question,
        )
        return run_agent(retriever_agent, task)

    def search_web(question: str) -> dict:
        # 1. Retrieve structured evidence from Tavily.
        evidence = web_retrieval_service.retrieve(question)

        # 2. Do not invoke the LLM if no evidence exists.
        if not evidence:
            return {
                "evidence": [],
                "agent_output": None,
            }

        # 3. Give the already-retrieved evidence to CrewAI.
        task = create_web_summary_task(
            agent=web_summary_agent,
            question=question,
            evidence=evidence,
        )

        # 4. The agent summarizes; it cannot call Tavily.
        agent_output = run_agent(
            web_summary_agent,
            task,
        )

        return {
            "evidence": evidence,
            "agent_output": agent_output,
        }


    return AgentOrchestrator(
        router=route_question,
        retriever=retrieve_question,
        web_search=search_web,
    )