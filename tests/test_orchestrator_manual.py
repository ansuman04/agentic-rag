
from src.orchestrator import build_local_orchestrator


def display_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print("QUESTION:", result.get("question"))
    print("ROUTE:", result.get("route"))
    print("STATUS:", result.get("status"))

    evidence = result.get("evidence", [])
    relevant = result.get("relevant_evidence", [])
    rejected = result.get("rejected_evidence", [])

    print("\n===== RETRIEVAL RESULTS =====")
    print("Total Retrieved:", len(evidence))
    print("Relevant Evidence:", len(relevant))
    print("Rejected Evidence:", len(rejected))

    print("\n===== RELEVANT EVIDENCE =====")

    for index, item in enumerate(relevant, start=1):
        print(f"\nSource {index}")

        if "source" in item:
            print("Document:", item["source"])
            print("Page:", item.get("page"))

        if "title" in item:
            print("Title:", item["title"])
            print("URL:", item.get("url"))

        text = item.get("text") or item.get("snippet", "")
        print("Text:", text[:400])

    print("\n===== GENERATED ANSWER =====")
    print(result.get("agent_output"))

    print("\n===== GROUNDING VALIDATION =====")

    grounding = result.get("grounding")

    if grounding is not None:
        print(
            "Grounding Status:",
            grounding.get("grounding_status"),
        )

        for index, claim in enumerate(
            grounding.get("claims", []),
            start=1,
        ):
            print(f"\nClaim {index}:")
            print("Claim:", claim.get("claim"))
            print("Verdict:", claim.get("verdict"))
            print("Reason:", claim.get("reason"))
    else:
        print("Grounding validation not available.")

    if result.get("error"):
        print("\n===== ERROR =====")
        print(result["error"])


def main():
    orchestrator = build_local_orchestrator()

    questions = [
        "Why do Transformers use positional encoding?",
        "What is the weather in Bengaluru today?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print("Processing question:", question)

        try:
            result = orchestrator.run(question)
            display_result(result)

        except Exception as exc:
            print(
                "\nPipeline execution failed:",
                type(exc).__name__,
                str(exc),
            )
            raise


if __name__ == "__main__":
    main()
