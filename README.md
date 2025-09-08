# 📚 PDF Ingestion and Semantic Search with LangChain and PostgreSQL (pgVector)

This project implements a **PDF ingestion** and **semantic search system** using **LangChain**, **PostgreSQL + pgVector**, and **OpenAI / Google Gemini models**.

The goal is to allow the user to **ask questions via CLI (command line interface)** and receive answers **based only on the ingested PDF content**, without using external knowledge.

---

## 🚀 Features

* 📥 **PDF Ingestion**:

  * Loads the document `document.pdf`.
  * Splits it into **chunks of 1000 characters with 150 overlap**.
  * Converts chunks into embeddings.
  * Stores vectors in PostgreSQL with **pgVector** extension.

* 🔎 **Semantic Search**:

  * User asks questions through the terminal.
  * The system retrieves the **10 most relevant chunks** (k=10).
  * Builds a prompt and calls an LLM (OpenAI or Gemini).
  * Ensures answers are based **only on the PDF**.

* ✅ **Rules**:

  * If the information is not in the PDF, it responds with:

    ```
    I don’t have the necessary information to answer your question.
    ```

---

## 🛠️ Tech Stack

* **Language**: Python
* **Framework**: LangChain
* **Database**: PostgreSQL + pgVector
* **Containerization**: Docker & Docker Compose

### Main Packages

* `langchain`
* `langchain_text_splitters`
* `langchain_openai`
* `langchain_google_genai`
* `langchain_postgres`
* `langchain_community`
* `psycopg2-binary`

---

## 📂 Project Structure

```
├── docker-compose.yml
├── requirements.txt      # Project dependencies
├── .env.example          # Template for environment variables
├── src/
│   ├── ingest.py         # PDF ingestion script
│   ├── search.py         # Semantic search script
│   ├── chat.py           # CLI interaction with user
├── document.pdf          # PDF to be ingested
└── README.md             # Setup & execution guide
```

---

## ⚙️ Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Copy `.env.example` to `.env` and add your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

---

## 🗄️ Start Database

The database uses **PostgreSQL + pgVector** via Docker Compose:

```bash
docker compose up -d
```

---

## 📥 Ingest PDF

Before running queries, you need to ingest the PDF:

```bash
python src/ingest.py
```

---

## 💬 Run Chat (CLI)

After ingestion, ask questions via terminal:

```bash
python src/chat.py
```

### Example:

**In-context question:**

```
QUESTION: What is the revenue of SuperTechIABrazil?
ANSWER: The revenue was 10 million reais.
```

**Out-of-context question:**

```
QUESTION: What is the capital of France?
ANSWER: I don’t have the necessary information to answer your question.
```

---

## 🧩 Notes

* Embedding models:

  * OpenAI → `text-embedding-3-small`
  * Gemini → `models/embedding-001`
* LLMs:

  * OpenAI → `gpt-5-nano`
  * Gemini → `gemini-2.5-flash-lite`
* The system is designed to **never hallucinate or generate external knowledge**.

---

## 📖 Useful Resources

* [LangChain Docs](https://python.langchain.com)
* [pgVector Extension](https://github.com/pgvector/pgvector)
* [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
* [Google Gemini API](https://ai.google.dev)

---

## 👨‍💻 Author

Developed for learning and practicing **RAG (Retrieval Augmented Generation)** with LangChain.