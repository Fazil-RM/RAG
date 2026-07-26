import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents

# Load environment variables
load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "vector_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile"
    ):

        # Initialize Vector Store
        self.vector_store = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model
        )

        # Check whether vector database already exists
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print("[INFO] Vector database not found. Building a new one...")

            docs = load_all_documents("data")
            self.vector_store.build_from_documents(docs)

        else:
            print("[INFO] Loading existing vector database...")
            self.vector_store.load()

        # Load Groq API Key
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please add it to your .env file."
            )

        # Initialize Groq LLM
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=llm_model,
            temperature=0.2
        )

        print(f"[INFO] Loaded Groq model: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:

        # Retrieve relevant chunks
        results = self.vector_store.query(query, top_k=top_k)

        texts = [
            r["metadata"]["text"]
            for r in results
            if r["metadata"] is not None
        ]

        if not texts:
            return "No relevant documents found."

        context = "\n\n".join(texts)

        prompt = f"""
You are an intelligent AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply exactly:

"I couldn't find this information in the uploaded documents."

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content


if __name__ == "__main__":

    rag = RAGSearch()

    while True:

        question = input("\nEnter your question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer = rag.search_and_summarize(question, top_k=5)

        print("\n" + "=" * 70)
        print("Answer:\n")
        print(answer)
        print("=" * 70)