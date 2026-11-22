from collections.abc import Callable
from src.document_retrieval import DocumentRetrievalService
from src.agents import create_document_summary_agent
from src.tasks import create_document_summary_task
from crewai import Crew, Process
import logging
from openai import InternalServerError
from src.relevance_grader import RelevanceGrader
from src.llm_relevance_scorer import LLMRelevanceScorer
from src.grounding_validator import GroundingValidator
from src.llm_grounding_validator import LLMGroundingScorer

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
        retriever: Callable[[str], dict],
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

        raise ValueError(f"Unsupported routing decision: {route!r}")


def build_retrieval_response(
    question: str,
    route: str,
    retrieval_result: dict,
) -> dict:
    evidence = retrieval_result["evidence"]
    relevant_evidence = retrieval_result.get("relevant_evidence", evidence)
    rejected_evidence = retrieval_result.get("rejected_evidence", [])
    grounding = retrieval_result.get("grounding")
    error = retrieval_result.get("error")

    if error and error.get("type") == "relevance_grading_error":
        status = "grading_failed"
    elif error and error.get("type") == "grounding_validation_error":
        status = "grounding_failed"
    elif error:
        status = "generation_failed"
    elif not evidence:
        status = "no_evidence"
    elif not relevant_evidence:
        status = "no_relevant_evidence"
    else:
        status = "retrieved"

    return {
        "question": question,
        "route": route,
        "status": status,
        "evidence": evidence,
        "relevant_evidence": relevant_evidence,
        "rejected_evidence": rejected_evidence,
        "agent_output": retrieval_result["agent_output"],
        "grounding": grounding,
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


def grade_and_summarize(
    question: str,
    evidence: list[dict],
    grader: RelevanceGrader,
    summary_agent,
    task_factory,
) -> dict:
    """
    Grade retrieved evidence and summarize relevant results.

    Preserve original evidence if grading or
    summarization fails.
    """

    if not evidence:
        return {
            "evidence": [],
            "relevant_evidence": [],
            "rejected_evidence": [],
            "agent_output": None,
            "error": None,
        }

    # Grade retrieved evidence.
    try:
        grading_result = grader.grade(
            question=question,
            evidence=evidence,
        )

    except (ValueError, InternalServerError) as exc:
        logger.warning(
            "Relevance grading failed: %s",
            exc,
        )

        return {
            "evidence": evidence,
            "relevant_evidence": [],
            "rejected_evidence": [],
            "agent_output": None,
            "error": {
                "type": "relevance_grading_error",
                "message": (
                    "Unable to grade retrieved evidence."
                ),
            },
        }

    relevant_evidence = grading_result[
        "relevant_evidence"
    ]

    rejected_evidence = grading_result[
        "rejected_evidence"
    ]

    # All evidence was graded irrelevant.
    if not relevant_evidence:
        return {
            "evidence": evidence,
            "relevant_evidence": [],
            "rejected_evidence": rejected_evidence,
            "agent_output": None,
            "error": None,
        }

    # Summarize only relevant evidence.
    task = task_factory(
        agent=summary_agent,
        question=question,
        evidence=relevant_evidence,
    )

    summary_result = run_summary_safely(
        agent=summary_agent,
        task=task,
        evidence=relevant_evidence,
    )

    return {
        "evidence": evidence,
        "relevant_evidence": relevant_evidence,
        "rejected_evidence": rejected_evidence,
        "agent_output": summary_result["agent_output"],
        "error": summary_result["error"],
    }

def validate_generated_answer(
    question: str,
    retrieval_result: dict,
    grounding_validator: GroundingValidator,
) -> dict:
    """
    Validate the generated answer against relevant evidence.

    Preserves the original retrieval result and answer.
    """

    result = retrieval_result.copy()
    result["grounding"] = None

    answer = result.get("agent_output")
    evidence = result.get("relevant_evidence", [])

    # Do not validate if an earlier stage failed.
    if result.get("error") is not None:
        return result

    # No answer or relevant evidence to validate.
    if not answer or not evidence:
        return result

    try:
        result["grounding"] = grounding_validator.validate(
            question=question,
            answer=answer,
            evidence=evidence,
        )

    except (ValueError, InternalServerError) as exc:
        logger.warning(
            "Answer grounding validation failed: %s",
            exc,
        )

        result["error"] = {
            "type": "grounding_validation_error",
            "message": "Unable to validate generated answer.",
        }

    return result



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

    relevance_scorer = LLMRelevanceScorer()

    relevance_grader = RelevanceGrader(
        scorer=relevance_scorer.score
    )
    # Initialize answer grounding verification.
    grounding_scorer = LLMGroundingScorer()

    grounding_validator = GroundingValidator(
        validator=grounding_scorer.score,
    )
    def route_question(question: str) -> str:
        task = create_routing_task(
            router_agent,
            question,
        )
        return run_agent(router_agent, task)

    
    def retrieve_question(question: str) -> dict:
        # Step 1: Retrieve documents from ChromaDB.
        evidence = document_retrieval_service.retrieve(
            question
        )

        # Step 2: Grade relevance and summarize.
        retrieval_result = grade_and_summarize(
            question=question,
            evidence=evidence,
            grader=relevance_grader,
            summary_agent=document_summary_agent,
            task_factory=create_document_summary_task,
        )

        # Step 3: Validate generated answer.
        return validate_generated_answer(
            question=question,
            retrieval_result=retrieval_result,
            grounding_validator=grounding_validator,
        )

 
    
    def search_web(question: str) -> dict:
        # Step 1: Retrieve web evidence from Tavily.
        evidence = web_retrieval_service.retrieve(
            question
        )

        # Step 2: Grade relevance and summarize.
        retrieval_result = grade_and_summarize(
            question=question,
            evidence=evidence,
            grader=relevance_grader,
            summary_agent=web_summary_agent,
            task_factory=create_web_summary_task,
        )

        # Step 3: Validate generated answer.
        return validate_generated_answer(
            question=question,
            retrieval_result=retrieval_result,
            grounding_validator=grounding_validator,
        )

    return AgentOrchestrator(
        router=route_question,
        retriever=retrieve_question,
        web_search=search_web,
    )
