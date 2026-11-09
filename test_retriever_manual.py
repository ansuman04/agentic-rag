
from crewai import Crew, Process

from src.agents import create_retriever_agent
from src.tasks import create_retrieval_task
from src.vector_store import VectorStore
from src.tools import DocumentSearchTool


def main():
    store = VectorStore()

    if store.collection.count() == 0:
        raise RuntimeError(
            "No indexed documents found. Run python main.py first."
        )

    tool = DocumentSearchTool(vector_store=store)

    agent = create_retriever_agent(tool)

    question = "Why do Transformers use positional encoding?"

    task = create_retrieval_task(
        agent=agent,
        question=question,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )

    result = crew.kickoff()

    print("\nRetrieved Evidence:")
    print(result.raw)


if __name__ == "__main__":
    main()
