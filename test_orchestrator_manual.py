
from unittest import result

from src.orchestrator import build_local_orchestrator


def main():
    orchestrator = build_local_orchestrator()

    questions = [
        "Why do Transformers use positional encoding?",
        "What is the weather in Bengaluru today?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print(f"Question: {question}")

        result = orchestrator.run(question)

        print(f"Selected Route: {result['route']}")

        if result["route"] == "websearch":
            print("\nOriginal Retrieved Evidence:")

            for index, item in enumerate(
                result["evidence"],
                start=1,
            ):
                print(f"\nSource {index}")
                print("Title:", item.get("title"))
                print("URL:", item.get("url"))
                print("Snippet:", item.get("snippet"))

            print("\nAgent Generated Output:")
            print(result["agent_output"])

        else:
            # Existing document retrieval output.
            print("\nRetrieved Evidence:")
            print(result["evidence"])


if __name__ == "__main__":
    main()