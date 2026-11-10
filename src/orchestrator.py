from collections.abc import Callable
from src.document_retrieval import DocumentRetrievalService
from src.agents import create_document_summary_agent
from src.tasks import create_document_summary_task
from crewai import Crew, Process
import logging
from openai import InternalServerError

logger = logging.getLogger(__name__)

from src.agents import (
    create_router_agent,
    create_retriever_agent,
    create_web_summary_agent,
)
from src.tasks import (
    create_routing_task,
    create_retrieval_task,
    create_web_summary_task,
)
from src.tools import DocumentSearchTool
from src.vector_store import VectorStore
from src.web_retrieval import WebRetrievalService
from src.web_tools import WebSearchTool


class AgentOrchestrator:
    """Route questions to document retrieval or web search."""

    def __init__(
        self,
        router: Callable[[str], str],
        retriever: Callable[[str], str],
        web_search: Callable[[str], dict] | None = None,
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
            document_result = self.retriever(question)

            return build_retrieval_response(
                question=question,
                route=route,
                retrieval_result=document_result,
            )


        if route == "websearch":
            if self.web_search is None:
                return {
                    "route": route,
                    "question": question,
                    "evidence": None,
                    "message": "Web search is not implemented yet.",
                }

            web_result = self.web_search(question)

            return build_retrieval_response(
                question=question,
                route=route,
                retrieval_result=web_result,
            )


        raise ValueError(
            f"Unsupported routing decision: {route!r}"
        )

def build_retrieval_response(
    question: str,
    route: str,
    retrieval_result: dict,
) -> dict:
    evidence = retrieval_result["evidence"]
    error = retrieval_result.get("error")

    if error:
        status = "generation_failed"
    elif evidence:
        status = "retrieved"
    else:
        status = "no_evidence"

    return {
        "question": question,
        "route": route,
        "status": status,
        "evidence": evidence,
        "agent_output": retrieval_result["agent_output"],
        "error": error,
    }


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

def run_summary_safely(
    agent,
    task,
    evidence: list[dict],
) -> dict:
    """
    Run the LLM summarizer while preserving retrieved evidence.

    Only recognized LLM server failures are recovered.
    Other exceptions propagate normally.
    """
    try:
        agent_output = run_agent(agent, task)

        return {
            "evidence": evidence,
            "agent_output": agent_output,
            "error": None,
        }

    except InternalServerError as exc:
        logger.warning(
            "LLM summarization failed after retrieval: %s",
            exc,
        )

        return {
            "evidence": evidence,
            "agent_output": None,
            "error": {
                "type": "llm_generation_error",
                "message": (
                    "The LLM server failed while "
                    "summarizing retrieved evidence."
                ),
            },
        }



def build_local_orchestrator() -> AgentOrchestrator:
    """Build the local Agentic RAG workflow."""

    store = VectorStore()
    document_retrieval_service = DocumentRetrievalService(
        vector_store=store
    )

    document_summary_agent = create_document_summary_agent()

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

    
    def retrieve_question(question: str) -> dict:
        # 1. Retrieve original document passages.
        evidence = document_retrieval_service.retrieve(
            question
        )

        # 2. Handle an empty knowledge base or no matches.
        if not evidence:
            return {
                "evidence": [],
                "agent_output": None,
            }

        # 3. Give the evidence to the CrewAI summarizer.
        task = create_document_summary_task(
            agent=document_summary_agent,
            question=question,
            evidence=evidence,
        )

        # 4. Generate a summary without searching again.
        return run_summary_safely(
            agent=document_summary_agent,
            task=task,
            evidence=evidence,
        )


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

        return run_summary_safely(
            agent=web_summary_agent,
            task=task,
            evidence=evidence,
        )


    return AgentOrchestrator(
        router=route_question,
        retriever=retrieve_question,
        web_search=search_web,
    )