
from src.document_loader import load_pdf
from src.vector_store import VectorStore
from src.rag_service import RAGService
from src.llm_client import LocalLLM

def main():
    pdf_path = "data/sample.pdf"

    print("Loading and indexing PDF...")
    records = load_pdf(pdf_path)

    store = VectorStore()
    store.add_documents(records)

    print("\nPDF indexed successfully.")

    llm = LocalLLM()

    rag = RAGService(
        vector_store=store,
        llm=llm,
    )

    question = "Why do Transformers use positional encoding?"

    print(f"\nQuestion: {question}")

    result = rag.answer(question)

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nRetrieved Sources:")

    for source in result["sources"]:
        print(
            f"{source['citation']} "
            f"(distance={source['distance']:.4f})"
        )


if __name__ == "__main__":
    main()