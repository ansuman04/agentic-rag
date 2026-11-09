
from src.document_loader import load_pdf
from src.vector_store import VectorStore


def main():
    pdf_path = "data/sample.pdf"

    print("Loading PDF...")
    records = load_pdf(
        pdf_path,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"\nTotal chunks: {len(records)}")

    vector_store = VectorStore()

    print("\nGenerating embeddings and storing metadata...")
    vector_store.add_documents(records)

    query = "Why do Transformers use positional encoding?"

    print(f"\nSearching for: {query}")

    results = vector_store.search(query, top_k=3)

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\n--- Result {index} ---")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Cosine distance: {result['distance']:.4f}")
        print(f"Text: {result['text'][:500]}")


if __name__ == "__main__":
    main()