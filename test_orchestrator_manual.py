
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

        print("Selected Route:", result["route"])
        print("Status:", result["status"])

        print("\nOriginal Retrieved Evidence:")

        for index, item in enumerate(
            result["evidence"],
            start=1,
        ):
            print(f"\nSource {index}")

            if result["route"] == "vectorstore":
                print("Document:", item["source"])
                print("Page:", item["page"])
                print("Distance:", item["distance"])
                print("Text:", item["text"])

            elif result["route"] == "websearch":
                print("Title:", item["title"])
                print("URL:", item["url"])
                print("Snippet:", item["snippet"])

        print("\nAgent Generated Output:")
        print(result["agent_output"])

if __name__ == "__main__":
    main()