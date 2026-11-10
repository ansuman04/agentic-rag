
from src.vector_store import VectorStore


class DocumentRetrievalService:
    """Retrieve structured document evidence without an LLM."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, question: str) -> list[dict]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        results = self.vector_store.search(
            query=question.strip(),
            top_k=5,
        )

        evidence = []

        for result in results:
            metadata = result["metadata"]

            evidence.append(
                {
                    "text": result["text"],
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_index": metadata.get(
                        "chunk_index"
                    ),
                    "document_id": metadata.get(
                        "document_id"
                    ),
                    "distance": result["distance"],
                }
            )

        return evidence