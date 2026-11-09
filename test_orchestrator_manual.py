
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

        if result["evidence"] is not None:
            print("\nRetrieved Evidence:")
            print(result["evidence"])
        else:
            print(result["message"])


if __name__ == "__main__":
    main()