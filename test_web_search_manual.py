
from crewai import Crew, Process

from src.agents import create_web_search_agent
from src.tasks import create_web_search_task
from src.web_tools import WebSearchTool


def main():
    tool = WebSearchTool()

    agent = create_web_search_agent(tool)

    question = "What is the latest stable Python release?"

    task = create_web_search_task(
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

    print("\n" + "=" * 70)
    print("Question:", question)
    print("\nWeb Search Evidence:")
    print(result.raw)


if __name__ == "__main__":
    main()
