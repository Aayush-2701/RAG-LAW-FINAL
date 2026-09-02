# RAG-LAW ⚖️ — AI Legal Assistant

> **Retrieval-Augmented Generation (RAG) over Indian and International legal documents — powered by Groq & LangChain.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

RAG-LAW is a production-ready **question-answering system** that lets you ask plain-language questions about legal documents and get accurate, grounded answers — with source citations, no hallucinations.

It combines:
- **FAISS** vector similarity search for fast document retrieval
- **HuggingFace sentence-transformers** for CPU-only embeddings (no GPU needed)
- **Groq** ultra-fast inference with multiple LLM choices
- **Streamlit** for an interactive, streaming chat interface

---

## 📄 Included Documents

| Document | Description |
|---|---|
| `Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf` | Indian Criminal Procedure Code (BNSS 2023) |
| `universal_declaration_of_human_rights.pdf` | UN Universal Declaration of Human Rights |

You can also **upload your own PDF** directly in the app.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    PDF["📄 PDF Documents\n(Root or Uploaded)"] --> Loader["PDFPlumberLoader"]
    Loader --> Splitter["RecursiveCharacterTextSplitter\nchunk=1000, overlap=200"]
    Splitter --> Embedder["all-MiniLM-L6-v2\n(HuggingFace, CPU-only)"]
    Embedder --> FAISS["FAISS Vector Store\n(pre-built in Vectorstore/)"]

    User["👤 User Query"] --> Search["Similarity Search\ntop-k = 5 chunks"]
    FAISS --> Search
    Search --> Context["Context + Citations"]
    Context --> Prompt["ChatPromptTemplate"]
    Prompt --> Groq["Groq LLM\n(streaming)"]
    Groq --> Answer["📝 Streamed Answer\n+ Source Citations"]
```

---

## ✨ Features

- 🔍 **Semantic search** — FAISS vector similarity, not keyword matching
- ⚡ **Streaming answers** — see the response token-by-token in real time
- 📄 **Source citations** — every answer shows the source document and page number
- 🤖 **Multiple LLM models** — switch between Groq models in the sidebar
- 📤 **Upload any PDF** — not just the preloaded documents
- 💬 **Chat history** — full conversation memory within a session
- 🚀 **Instant cold starts** — pre-built FAISS index committed to the repo
- 🎨 **Dark theme** — readable, distraction-free UI

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A **free** Groq API key from [console.groq.com](https://console.groq.com)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Aayush-2701/RAG-LAW-FINAL.git
cd RAG-LAW-FINAL

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and replace the placeholder with your real key

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Community Cloud

1. **Fork** or push this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and set **Main file path** to `app.py`.
4. Click **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your-groq-api-key-here"
   ```
5. Click **Deploy**. The app will be live in ~2 minutes.

> **No GPU required.** Embeddings run on CPU via `sentence-transformers`. The pre-built FAISS index in `Vectorstore/` is loaded instantly — no rebuild on cold start.

---

## 📁 Project Structure

```
RAG-LAW-FINAL/
├── app.py                                        # Streamlit web application
├── rag_pipeline.py                               # Core RAG backend
├── requirements.txt                              # Python dependencies
├── runtime.txt                                   # Python 3.11 spec
├── packages.txt                                  # OS-level deps (Streamlit Cloud)
├── .gitignore
├── .streamlit/
│   ├── config.toml                               # Dark theme + UI config
│   └── secrets.toml.example                      # API key template
├── Vectorstore/
│   ├── index.faiss                               # Pre-built vector index
│   └── index.pkl                                 # Index metadata
├── Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf
└── universal_declaration_of_human_rights.pdf
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **UI / Framework** | [Streamlit](https://streamlit.io) |
| **LLM** | [Groq](https://groq.com) — `deepseek-r1-distill-llama-70b` (default) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (CPU) |
| **Orchestration** | [LangChain](https://langchain.com) 0.3.x |
| **PDF Parsing** | pdfplumber + pypdf |

---

## 🧩 Use the RAG Pipeline Directly

```python
from rag_pipeline import build_preloaded_index, answer_query

# Load the preloaded index (uses pre-built Vectorstore/ for speed)
vs = build_preloaded_index()

# Stream an answer
stream_gen, citations = answer_query(vs, "What are the punishment provisions for theft?")
for token in stream_gen:
    print(token, end="", flush=True)

# Print citations
for c in citations:
    print(f"\n[{c['source']}, p.{c['page']}]: {c['snippet']}")
```

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It is **not** a substitute for qualified legal advice. Always consult a licensed legal professional for legal matters.

---

## 🤝 Contributing

Pull requests are welcome! Please:
1. Fork the repo and create a feature branch.
2. Make your changes with clear commit messages.
3. Open a PR with a description of what changed and why.

---

## 📜 License

MIT © [Aayush-2701](https://github.com/Aayush-2701)
