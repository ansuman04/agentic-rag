
import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, db_path="./chroma_db"):
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name="agentic_rag_documents",
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add_documents(self, chunks):
        if not chunks:
            raise ValueError("No chunks to store")

        embeddings = self.model.encode(chunks).tolist()

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
        )

        print(f"Stored {len(chunks)} chunks in ChromaDB")

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query]).tolist()

        count = self.collection.count()

        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, count),
            include=["documents", "distances"],
        )

        return list(
            zip(
                results["documents"][0],
                results["distances"][0],
            )
        )