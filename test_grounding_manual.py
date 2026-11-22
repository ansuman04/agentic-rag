
from src.grounding_validator import GroundingValidator
from src.llm_grounding_validator import LLMGroundingScorer


def main():
    question = "Explain Transformer positional encoding."

    answer = (
        "Transformers use sinusoidal positional encodings "
        "to represent token positions. "
        "The Transformer architecture was introduced in 2020."
    )

    evidence = [
        {
            "text": (
                "The Transformer architecture was "
                "introduced in the 2017 paper "
                "'Attention Is All You Need'. "
                "The model uses sine and cosine functions "
                "of different frequencies for positional "
                "encodings, which are added to input "
                "embeddings."
            ),
            "source": "sample.pdf",
            "page": 6,
        }
    ]

    scorer = LLMGroundingScorer()

    validator = GroundingValidator(
        validator=scorer.score,
    )

    result = validator.validate(
        question=question,
        answer=answer,
        evidence=evidence,
    )

    print("\nGrounding Status:")
    print(result["grounding_status"])

    print("\nClaim Assessments:")

    for claim in result["claims"]:
        print("\nClaim:", claim["claim"])
        print("Verdict:", claim["verdict"])
        print("Reason:", claim["reason"])


if __name__ == "__main__":
    main()
