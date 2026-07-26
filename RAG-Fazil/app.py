from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


if __name__ == "__main__":

    # Load documents
    docs = load_all_documents("data")

    # Create Vector Store
    store = FaissVectorStore("vector_db")

    # Build vector database only if it doesn't exist
    # Uncomment this ONLY for the first run
    store.build_from_documents(docs)

    # Load existing vector database
    # Uncomment this AFTER the first run
    #store.load()

    # Create RAG
    rag = RAGSearch()

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        answer = rag.search_and_summarize(query, top_k=5)

        print("\n==============================")
        print("Answer:\n")
        print(answer)
        print("==============================")