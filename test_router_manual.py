
from crewai import Crew, Process

from src.agents import create_router_agent
from src.tasks import create_routing_task


def route_question(question: str) -> str:
    agent = create_router_agent()

    task = create_routing_task(
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
    route = result.raw.strip().lower()

    if route not in {"vectorstore", "websearch"}:
        raise ValueError(
            f"Unexpected router response: {route}"
        )

    return route


if __name__ == "__main__":
    questions = [
        "What is self-attention?",
        "Why do Transformers use positional encoding?",
        "What is the weather in Bengaluru today?",
        "What are the latest developments in AI?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print(f"Route: {route_question(question)}")