from typing import List, Any
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)

        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        print(
            f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks"
        )

        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Create embeddings for all chunks.
        """

        texts = [chunk.page_content for chunk in chunks]

        print(f"[INFO] Creating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        print(f"[INFO] Embeddings shape: {embeddings.shape}")

        return embeddings.astype(np.float32)


# Example Usage
if __name__ == "__main__":

    docs = load_all_documents("data")

    emb_pipe = EmbeddingPipeline()

    chunks = emb_pipe.chunk_documents(docs)

    embeddings = emb_pipe.embed_chunks(chunks)

    print("\nFirst Embedding:\n")
    print(embeddings[0] if len(embeddings) > 0 else None)