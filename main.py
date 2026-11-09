
from src.document_loader import load_pdf, split_text
from src.vector_store import VectorStore


def main():
    pdf_path = "data/sample.pdf"

    print("Loading PDF...")
    text = load_pdf(pdf_path)

    chunks = split_text(
        text,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"\nTotal chunks: {len(chunks)}")

    vector_store = VectorStore()

    print("\nGenerating embeddings...")
    vector_store.add_documents(chunks)

    query = "What does an agent do in Agentic RAG?"

    print(f"\nSearching for: {query}")

    results = vector_store.search(query, top_k=3)

    for index, (document, distance) in enumerate(
        results, start=1
    ):
        print(f"\nResult {index}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Text: {document}")


if __name__ == "__main__":
    main()