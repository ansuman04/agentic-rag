
import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, db_path="./chroma_db"):
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name="agentic_rag_documents_v2",
            configuration={"hnsw": {"space": "cosine"}},
        )

    
    def add_documents(self, records):
        """Replace existing chunks for a source and index its current version."""
        if not records:
            raise ValueError("No records to store")

        sources = {record["metadata"]["source"] for record in records}

        if len(sources) != 1:
            raise ValueError(
                "Index one source document at a time"
            )

        source = next(iter(sources))
        document_ids = {
            record["metadata"]["document_id"] for record in records
        }

        if len(document_ids) != 1:
            raise ValueError(
                "All chunks must belong to the same document version"
            )

        documents = [record["text"] for record in records]
        ids = [record["id"] for record in records]
        metadatas = [record["metadata"] for record in records]

        # Generate embeddings before changing the existing collection.
        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
        ).tolist()

        # Remove the previous version of this source.
        existing = self.collection.get(
            where={"source": source},
            include=[],
        )

        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        # Store the new version with its metadata.
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"Indexed {len(records)} chunks for {source}")

    def search(self, query, top_k=3):
        """Retrieve chunks with their source metadata."""
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        count = self.collection.count()

        if count == 0:
            return []

        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, count),
            include=["documents", "distances", "metadatas"],
        )

        return [
            {
                "text": document,
                "distance": distance,
                "metadata": metadata,
            }
            for document, distance, metadata in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]