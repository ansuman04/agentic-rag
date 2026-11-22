
from src.answer_corrector import AnswerCorrector
from src.grounding_validator import GroundingValidator
from src.llm_grounding_validator import LLMGroundingScorer
from src.llm_answer_corrector import LLMAnswerCorrector


def main():
    question = "When was the Transformer architecture introduced?"

    original_answer = (
        "The Transformer architecture was introduced in 2020."
    )

    evidence = [
        {
            "text": (
                "The Transformer architecture was introduced "
                "in the 2017 paper Attention Is All You Need."
            ),
            "source": "test_evidence",
            "page": 1,
        }
    ]

    grounding_scorer = LLMGroundingScorer()

    validator = GroundingValidator(
        validator=grounding_scorer.score
    )

    llm_corrector = LLMAnswerCorrector()

    corrector = AnswerCorrector(
        corrector=llm_corrector.correct,
        grounding_validator=validator,
    )

    # First grounding check.
    initial_grounding = validator.validate(
        question=question,
        answer=original_answer,
        evidence=evidence,
    )

    print("\nInitial Grounding:")
    print(initial_grounding["grounding_status"])

    # Correct only if needed, then validate again.
    result = corrector.correct(
        question=question,
        answer=original_answer,
        evidence=evidence,
        initial_grounding=initial_grounding,
    )

    print("\n===== ANSWER CORRECTION RESULTS =====")
    print("Original Answer:", result["original_answer"])
    print("Corrected Answer:", result["corrected_answer"])
    print("Verified Answer:", result["verified_answer"])
    print("Correction Status:", result["correction_status"])
    print("Attempts:", result["attempts"])
    print(
        "Final Grounding:",
        result["grounding"]["grounding_status"],
    )


if __name__ == "__main__":
    main()
