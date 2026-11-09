
import json
from unittest.mock import Mock

from src.tools import DocumentSearchTool


def test_document_search_returns_evidence():
    store = Mock()

    store.search.return_value = [
        {
            "text": (
                "Transformers use positional encoding "
                "to represent sequence order."
            ),
            "distance": 0.21,
            "metadata": {
                "source": "sample.pdf",
                "page": 6,
            },
        }
    ]

    tool = DocumentSearchTool(vector_store=store)

    output = tool._run(
        question="Why use positional encoding?"
    )

    evidence = json.loads(output)

    assert len(evidence) == 1
    assert evidence[0]["source"] == "sample.pdf"
    assert evidence[0]["page"] == 6
    assert evidence[0]["distance"] == 0.21

    store.search.assert_called_once_with(
        query="Why use positional encoding?",
        top_k=5,
    )
