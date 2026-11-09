
from crewai import Task, Agent


def create_routing_task(
    agent: Agent,
    question: str,
) -> Task:
    return Task(
        description=(
            f"Classify this question:\n{question}\n\n"
            "Choose exactly one route:\n"
            "vectorstore - for questions about the Transformer "
            "research paper and its concepts.\n"
            "websearch - for current events, live data, or "
            "information outside the document.\n\n"
            "Do not answer the question. "
            "Return only the route name."
        ),
        expected_output=(
            "Exactly one word: vectorstore or websearch."
        ),
        agent=agent,
    )


def create_retrieval_task(
    agent: Agent,
    question: str,
) -> Task:
    return Task(
        description=(
            f"Retrieve information to answer:\n{question}\n\n"
            "You MUST use the document_search tool. "
            "Retrieve relevant passages from the indexed PDF. "
            "Return the evidence with its source filename "
            "and page number. Do not invent information."
        ),
        expected_output=(
            "Relevant document passages with filename "
            "and page references."
        ),
        agent=agent,
    )

