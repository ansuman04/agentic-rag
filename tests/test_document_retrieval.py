
from unittest.mock import Mock

import pytest

from src.document_retrieval import DocumentRetrievalService


def test_document_retrieval_preserves_evidence():
    original_results = [
        {
            "text": "Transformers use positional encoding.",
            "metadata": {
                "source": "sample.pdf",
                "page": 6,
                "chunk_index": 2,
                "document_id": "document-123",
            },
            "distance": 0.12,
        }
    ]

    mock_store = Mock()
    mock_store.search.return_value = original_results

    service = DocumentRetrievalService(
        vector_store=mock_store
    )

    evidence = service.retrieve(
        "Why do Transformers use positional encoding?"
    )

    assert evidence == [
        {
            "text": "Transformers use positional encoding.",
            "source": "sample.pdf",
            "page": 6,
            "chunk_index": 2,
            "document_id": "document-123",
            "distance": 0.12,
        }
    ]

    mock_store.search.assert_called_once_with(
        query="Why do Transformers use positional encoding?",
        top_k=5,
    )


def test_document_retrieval_empty_results():
    mock_store = Mock()
    mock_store.search.return_value = []

    service = DocumentRetrievalService(mock_store)

    assert service.retrieve("Unknown question") == []


def test_document_retrieval_rejects_empty_question():
    mock_store = Mock()

    service = DocumentRetrievalService(mock_store)

    with pytest.raises(ValueError):
        service.retrieve("   ")

    mock_store.search.assert_not_called()
