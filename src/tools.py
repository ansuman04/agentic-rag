
import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.vector_store import VectorStore


class DocumentSearchInput(BaseModel):
    question: str = Field(
        ...,
        description="Question to search in the PDF knowledge base",
    )


class DocumentSearchTool(BaseTool):
    name: str = "document_search"
    description: str = (
        "Search the locally indexed PDF knowledge base "
        "using semantic similarity. Returns relevant passages "
        "with source filenames, page numbers and distances."
    )
    args_schema: type[BaseModel] = DocumentSearchInput

    def __init__(self, vector_store: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self._vector_store = vector_store

    def _run(self, question: str) -> str:
        results = self._vector_store.search(
            query=question,
            top_k=5,
        )

        evidence = [
            {
                "text": result["text"],
                "source": result["metadata"]["source"],
                "page": result["metadata"]["page"],
                "distance": result["distance"],
            }
            for result in results
        ]

        return json.dumps(evidence, ensure_ascii=False)