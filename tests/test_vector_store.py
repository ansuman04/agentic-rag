
from src.document_loader import load_pdf
from src.vector_store import VectorStore


def test_reindex_same_document(tmp_path, sample_pdf):
    records = load_pdf(str(sample_pdf))

    store = VectorStore(db_path=str(tmp_path / "chroma"))

    store.add_documents(records)
    first_count = store.collection.count()

    store.add_documents(records)
    second_count = store.collection.count()

    assert first_count == len(records)
    assert second_count == first_count


def test_search_returns_metadata(tmp_path, sample_pdf):
    records = load_pdf(str(sample_pdf))

    store = VectorStore(db_path=str(tmp_path / "chroma"))

    store.add_documents(records)

    results = store.search(
        "Why do Transformers use positional encoding?",
        top_k=3,
    )

    assert len(results) == min(3, len(records))

    for result in results:
        assert "text" in result
        assert "distance" in result
        assert "metadata" in result
        assert "source" in result["metadata"]
        assert "page" in result["metadata"]
        assert "document_id" in result["metadata"]


def test_replace_modified_document(tmp_path):
    store = VectorStore(db_path=str(tmp_path / "chroma"))

    original = [{
        "id": "version1_chunk0",
        "text": "The original document discusses Transformers.",
        "metadata": {
            "source": "sample.pdf",
            "page": 1,
            "chunk_index": 0,
            "document_id": "version1",
        },
    }]

    updated = [{
        "id": "version2_chunk0",
        "text": "The updated document discusses positional encoding.",
        "metadata": {
            "source": "sample.pdf",
            "page": 1,
            "chunk_index": 0,
            "document_id": "version2",
        },
    }]

    store.add_documents(original)
    assert store.collection.count() == 1

    store.add_documents(updated)

    assert store.collection.count() == 1

    stored = store.collection.get(
        where={"source": "sample.pdf"},
        include=["documents", "metadatas"],
    )

    assert stored["ids"] == ["version2_chunk0"]
    assert stored["documents"] == [
        "The updated document discusses positional encoding."
    ]
    assert stored["metadatas"][0]["document_id"] == "version2"