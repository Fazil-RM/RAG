# 🚀 RAG Assistant V1

An end-to-end Retrieval-Augmented Generation (RAG) system built using Python, LangChain, FAISS, and Google Gemini.

This project loads documents, converts them into semantic embeddings, stores them in a vector database, retrieves the most relevant information for a query, and generates context-aware answers using an LLM.

---

## 📌 Features

- 📄 PDF document loading
- ✂️ Intelligent text chunking
- 🧠 Semantic embeddings
- 📦 FAISS vector database
- 🔎 Semantic similarity search
- 🤖 Context-aware answer generation
- ⚡ Automatic vector database creation
- 📁 Multiple document format support
  - PDF
  - TXT
  - CSV
  - DOCX
  - XLSX
  - JSON

---

# 📂 Project Structure

```text
RAG-Assistant/
│
├── data/
│   └── Documents
│
├── src/
│   ├── data_loader.py
│   ├── embedding.py
│   ├── vectorstore.py
│   ├── search.py
│   └── __init__.py
│
├── vector_db/
│   ├── faiss.index
│   └── metadata.pkl
│
├── app.py
├── config.py
├── requirements.txt
└── .env
```

---

# ⚙️ Pipeline

```text
Documents
     │
     ▼
Document Loader
     │
     ▼
Text Chunking
     │
     ▼
Embedding Generation
     │
     ▼
FAISS Vector Store
     │
     ▼
Similarity Search
     │
     ▼
Context Retrieval
     │
     ▼
LLM (Gemini)
     │
     ▼
Generated Answer
```

---

# 🛠 Tech Stack

- Python
- LangChain
- FAISS
- Sentence Transformers
- Google Gemini API
- NumPy
- Pickle
- dotenv

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/RAG-Assistant.git

cd RAG-Assistant
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

---

# ▶️ Run the Project

```bash
python app.py
```

Example

```
Enter your question:

What is Retrieval-Augmented Generation?

Answer:

Retrieval-Augmented Generation (RAG) combines document retrieval with Large Language Models to generate context-aware responses...
```

---

# 📁 Supported File Types

| Format | Supported |
|---------|-----------|
| PDF |  ✅ |
| TXT |  ✅ |
| CSV |  ✅ |
| DOCX | ✅ |
| XLSX | ✅ |
| JSON | ✅ |

---

# 📖 How It Works

1. Load documents from the `data` folder.
2. Split documents into semantic chunks.
3. Generate embeddings for every chunk.
4. Store embeddings in FAISS.
5. Convert the user query into an embedding.
6. Retrieve the most relevant chunks.
7. Pass the retrieved context to Gemini.
8. Generate a final answer.

---

# 📸 Architecture

```
User Question
      │
      ▼
Query Embedding
      │
      ▼
FAISS Search
      │
      ▼
Top K Chunks
      │
      ▼
Prompt Builder
      │
      ▼
Gemini LLM
      │
      ▼
Final Response
```

---

# 📈 Future Improvements

- 💬 Conversational Memory
- 🌐 Web Interface
- 📑 Source Citations
- 📚 Multi-document Collections
- 🔄 Incremental Document Updates
- ☁️ Cloud Deployment
- 📊 Hybrid Search
- 🚀 Better Embedding Models
- 🔐 User Authentication
- 📄 Upload Documents from UI

---

# 🎯 Learning Outcomes

This project helped me understand:

- Retrieval-Augmented Generation (RAG)
- LangChain pipelines
- Document chunking strategies
- Embedding generation
- Vector databases
- Semantic search
- Prompt engineering
- LLM integration
- End-to-end AI application development

---

# 🤝 Contributing

Contributions, improvements, and suggestions are always welcome.

Feel free to fork the repository and submit a pull request.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Mohamed Fazil RM**

- GitHub: https://github.com/Fazil-RM
- LinkedIn: https://linkedin.com/in/your-linkedin-profile

---

⭐ If you found this project useful, consider giving it a star!
