import os
import faiss
import numpy as np
import pickle
from typing import List, Any

from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "vector_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        """
        Build FAISS vector database from raw documents.
        """

        print(f"[INFO] Building vector store from {len(documents)} documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page", None),
            }
            for chunk in chunks
        ]

        self.add_embedding(
            np.array(embeddings).astype("float32"),
            metadatas,
        )

        self.save()

        print(f"[INFO] Vector store built and saved to '{self.persist_dir}'")

    def add_embedding(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        """
        Add embeddings into FAISS index.
        """

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS index")

    def save(self):
        """
        Save FAISS index and metadata.
        """

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved FAISS index and metadata to '{self.persist_dir}'")

    def load(self):
        """
        Load FAISS index and metadata.
        """

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded FAISS index and metadata from '{self.persist_dir}'")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Search the FAISS index using an embedding.
        """

        if self.index is None:
            raise ValueError(
                "Vector database not loaded. Call load() or build_from_documents() first."
            )

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None

            results.append(
                {
                    "index": idx,
                    "distance": float(dist),
                    "metadata": meta,
                }
            )

        return results

    def query(self, query_text: str, top_k: int = 5):
        """
        Convert question into embedding and search.
        """

        print(f"[INFO] Query: {query_text}")

        query_embedding = self.model.encode(
            [query_text]
        ).astype("float32")

        return self.search(query_embedding, top_k=top_k)


# Example Usage
if __name__ == "__main__":

    from src.data_loader import load_all_documents

    docs = load_all_documents("data")

    store = FaissVectorStore("vector_db")

    store.build_from_documents(docs)

    store.load()

    results = store.query("What is attention mechanism?", top_k=3)

    print(results)