
from src.web_retrieval import WebRetrievalService
from src.web_tools import WebSearchTool


def main():
    service = WebRetrievalService(
        tool=WebSearchTool()
    )

    question = "What is the weather in Bengaluru today?"

    evidence = service.retrieve(question)

    print("\nQuestion:", question)
    print("Retrieved results:", len(evidence))

    for index, item in enumerate(evidence, start=1):
        print(f"\nResult {index}")
        print("Title:", item.get("title"))
        print("URL:", item.get("url"))
        print("Snippet:", item.get("snippet"))


if __name__ == "__main__":
    main()