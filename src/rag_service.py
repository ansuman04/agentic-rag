
from src.llm_client import LocalLLM
from src.vector_store import VectorStore


class RAGService:
    def __init__(self, vector_store: VectorStore, llm: LocalLLM):
        self.vector_store = vector_store
        self.llm = llm

    def answer(self, question: str, top_k: int = 5) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        results = self.vector_store.search(
            query=question,
            top_k=top_k,
        )

        if not results:
            return {
                "answer": "No indexed documents are available.",
                "sources": [],
            }

        context_parts = []
        sources = []

        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]

            citation = (
                f"[{index}] {metadata['source']}, "
                f"page {metadata['page']}"
            )

            context_parts.append(
                f"{citation}\n{result['text']}"
            )

            sources.append({
                "citation": citation,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": result["distance"],
            })

        context = "\n\n".join(context_parts)

        prompt = f"""
Answer the question using ONLY the provided context.

If the context does not contain enough information,
respond with:
"I don't have enough information in the documents."

Include citations like [1] or [2] when making claims.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": sources,
        }