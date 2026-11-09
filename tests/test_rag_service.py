
from unittest.mock import Mock

import pytest

from src.rag_service import RAGService


@pytest.fixture
def rag_service():
    vector_store = Mock()
    llm = Mock()

    service = RAGService(
        vector_store=vector_store,
        llm=llm,
    )

    return service, vector_store, llm


def test_answer_with_retrieved_context(rag_service):
    service, vector_store, llm = rag_service

    vector_store.search.return_value = [
        {
            "text": (
                "Transformers use positional encoding "
                "to represent token order."
            ),
            "distance": 0.21,
            "metadata": {
                "source": "sample.pdf",
                "page": 6,
            },
        }
    ]

    llm.generate.return_value = (
        "Positional encoding represents token order [1]."
    )

    result = service.answer(
        "Why use positional encoding?"
    )

    assert result["answer"] == (
        "Positional encoding represents token order [1]."
    )

    assert len(result["sources"]) == 1
    assert result["sources"][0]["page"] == 6

    vector_store.search.assert_called_once_with(
        query="Why use positional encoding?",
        top_k=5,
    )

    llm.generate.assert_called_once()

    prompt = llm.generate.call_args.args[0]

    assert "Transformers use positional encoding" in prompt
    assert "[1] sample.pdf, page 6" in prompt


def test_empty_question_raises_error(rag_service):
    service, vector_store, llm = rag_service

    with pytest.raises(ValueError):
        service.answer("   ")

    vector_store.search.assert_not_called()
    llm.generate.assert_not_called()


def test_no_retrieved_documents(rag_service):
    service, vector_store, llm = rag_service

    vector_store.search.return_value = []

    result = service.answer(
        "What is positional encoding?"
    )

    assert result["answer"] == (
        "No indexed documents are available."
    )

    assert result["sources"] == []

    llm.generate.assert_not_called()