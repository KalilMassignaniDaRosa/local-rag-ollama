
# 🚀 Advanced Local RAG System

A sophisticated **Retrieval-Augmented Generation (RAG)** system that runs entirely on your local machine using **Ollama**, **PostgreSQL with pgvector**, and **advanced document processing**.

---

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose  
- Python **3.8+**  
- Ollama (with at least one model installed)

---

## 1. Clone and Setup

```bash
git clone <repository-url>
cd rag-local
```

---

## 2. Start Services with Docker

```bash
docker-compose up -d
```

This starts:

- **PostgreSQL with pgvector** on port **5433**  
- **Ollama** on port **11434**

---

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Pull Required Models

```bash
# Pull LLM model
ollama pull llama3.2:3b
```

---

## 5. Run the System

```bash
python cli.py
```

---

# 🔄 Basic Workflow

- **Place Documents:** Add PDF files to the `documents/` folder  
- **Ingest Documents:** Use option **1** in the CLI to load documents  
- **Process & Index:** Use option **2** to create chunks and embeddings  
- **Query:** Use options **3** or **4** to ask questions  

---

# 🧠 Advanced Query Features

## Template Types
- `query` — Standard query with precise citations  
- `summary` — Structured document summarization  
- `comparison` — Comparative analysis between documents  
- `extraction` — Specific information extraction  
- `qa` — Optimized question-answering  

## Output Formats
- `text` — Human-readable text format  
- `json` — Structured JSON response  
- `markdown` — Markdown formatted response

---

# ⚙️ Performance Tuning

## Embedding Generation
- Use **all-MiniLM-L6-v2** for a balance of speed and quality.  
- Consider **paraphrase-multilingual-MiniLM-L12-v2** for multilingual content.

## Chunking Strategy
- **Fixed:** Fastest, good for uniform documents.  
- **Semantic:** Better for structured documents with clear paragraphs.  
- **Sentence:** Best for question-answer pairs and precise retrieval.

## Search Parameters
- **Threshold 0.2–0.3:** Higher recall, more results.  
- **Threshold 0.4–0.6:** Balanced precision and recall.  
- **Threshold 0.7+:** High precision, fewer results.

---

# 🧩 Configuration & Files
- `docker-compose.yml` — service definitions for PostgreSQL + pgvector and Ollama  
- `requirements.txt` — Python dependencies  
- `cli.py` — command-line interface for ingestion and querying  
- `documents/` — place PDFs here for ingestion  
- Database: PostgreSQL with `pgvector` extension (port **5433**)  
- Ollama API: default **http://localhost:11434**

---

# 🙏 Acknowledgments
- **Ollama** — local LLM hosting  
- **PostgreSQL & pgvector** — efficient vector storage and search  
- **Sentence Transformers** — embedding models  
- **PyPDF2** — PDF text extraction