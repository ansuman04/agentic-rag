
from src.llm_relevance_scorer import LLMRelevanceScorer
from src.relevance_grader import RelevanceGrader


def main():
    question = "Why do Transformers use positional encoding?"

    evidence = [
        {
            "text": (
                "Positional encodings provide information "
                "about the position of tokens in a sequence."
            ),
            "source": "sample.pdf",
            "page": 6,
        },
        {
            "text": (
                "The model was trained using eight GPUs "
                "for several hours."
            ),
            "source": "sample.pdf",
            "page": 10,
        },
        {
            "text": (
                "Sinusoidal positional encodings allow "
                "the model to learn relative positions."
            ),
            "source": "sample.pdf",
            "page": 6,
        },
    ]

    scorer = LLMRelevanceScorer()
    grader = RelevanceGrader(scorer=scorer.score)

    print("\nQuestion:", question)
    print("\nRunning local Qwen2.5 relevance grading...\n")

    result = grader.grade(question, evidence)

    print("\n========== RELEVANT EVIDENCE ==========")

    for item in result["relevant_evidence"]:
        print(f"Page {item['page']}: {item['text']}")

    print("\n========== REJECTED EVIDENCE ==========")

    for item in result["rejected_evidence"]:
        print(f"Page {item['page']}: {item['text']}")

    print("\n========== SUMMARY ==========")
    print(
        "Relevant:",
        len(result["relevant_evidence"]),
    )
    print(
        "Rejected:",
        len(result["rejected_evidence"]),
    )


if __name__ == "__main__":
    main()