1.  What is RAG-LAW?

RAG-LAW is a Retrieval-Augmented Generation (RAG) application built specifically for legal documents. Instead of relying on an LLM's training knowledge (which can be outdated or hallucinated), it retrieves the most relevant passages directly from your documents and grounds every answer in those passages.
The three-step pipeline:
•	Split — Splits your PDF into text chunks and stores them in a FAISS vector index
•	Retrieve — Retrieves the most semantically relevant passages for your question
•	Answer — Answers using only those passages — grounded, not guessed

2.  Features

Feature	Description
📄  Upload any legal PDF	Contracts, legislation, case law, treaties — any PDF works
📚  Bundled sample documents	BNSS 2023 & UDHR included — ready to query out of the box
🔍  Semantic search	FAISS + all-MiniLM-L6-v2 finds relevant passages even with different wording
🤖  DeepSeek R1 via Groq	Fast, accurate LLM answers grounded strictly in document context
📑  Source transparency	Every answer shows the exact retrieved passages it was based on
⚡  Smart caching	Embedding model and index loaded once per session — no reprocessing
🚫  Hallucination guard	Model says "I don't know" rather than making things up
☁️  Zero local dependencies	Deploys on Streamlit Community Cloud with one click

3.  Live Demo

🔗  Live App:  https://rag-law-kz952wqnztdqebsx3ormv7.streamlit.app/
     (Replace with your own URL after deploying)

4.  Architecture

app.py  (Streamlit UI)
   ├── 📂 Preloaded Documents Tab  (cached FAISS index)
   └── 📤 Upload PDF Tab           (per-session index)
              │
              ▼
rag_pipeline.py  (Backend)
   ├── HuggingFace Embeddings  (all-MiniLM-L6-v2, CPU-only)
   ├── FAISS Vector Index  →  Similarity Search
   └── ChatGroq  (deepseek-r1-distill-llama-70b)
              │
              ▼
   Grounded Answer  +  Source Citations

Layer	Technology
UI Framework	Streamlit 1.42
Orchestration	LangChain 0.3
Embeddings	sentence-transformers / all-MiniLM-L6-v2  (CPU, free)
Vector Store	FAISS-CPU 1.9
LLM	deepseek-r1-distill-llama-70b  via Groq API
PDF Parsing	pdfplumber

5.  Deploy to Streamlit Community Cloud

Step	Action
Step 1	Fork this repository to your GitHub account
Step 2	Get a free Groq API key at  console.groq.com  (takes 30 seconds)
Step 3	Go to share.streamlit.io → New App → fill in Repository, Branch: main, Main file path: app.py
Step 4	Open Advanced settings → Secrets and paste:   GROQ_API_KEY = "your-key-here"
Step 5	Click Deploy 🎉

⏱  First load takes ~30–60 seconds while the embedding model downloads and PDFs are indexed. All subsequent visits in the same session are instant.

6.  Run Locally

Prerequisites
•	Python 3.11+
•	A free Groq API key — get one at console.groq.com

Setup Commands
# 1. Clone
git clone https://github.com/Aayush-2701/RAG-LAW.git
cd RAG-LAW

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows PowerShell
source .venv/bin/activate      # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
# Edit secrets.toml and paste your GROQ_API_KEY

# 5. Run
streamlit run app.py

Adding Your Own PDFs
Drop any PDF into the pdfs/ folder. It will be automatically detected and indexed the next time the app starts.

7.  Dependencies

Package	Version	Purpose
streamlit	1.42.0	Web UI framework
langchain	0.3.17	RAG orchestration
langchain-community	0.3.16	FAISS integration
langchain-groq	0.2.4	Groq LLM connector
langchain-huggingface	0.1.2	HuggingFace embeddings connector
sentence-transformers	3.4.1	all-MiniLM-L6-v2 embedding model
faiss-cpu	1.9.0.post1	Vector similarity search (CPU)
pdfplumber	0.11.4	PDF text extraction
groq	0.16.0	Groq SDK

8.  Security & Secrets

•	Single secret — GROQ_API_KEY is the only secret required
•	Source — Read from st.secrets (Streamlit Cloud) or os.environ (local dev)
•	No leaks — Never hardcoded anywhere in the codebase
•	Git-safe — .streamlit/secrets.toml is git-ignored and never committed

secrets.toml format
GROQ_API_KEY = "your-groq-api-key-here"

9.  Bundled Sample Documents

Document	Description
Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf	Indian Code of Criminal Procedure (BNSS 2023)
universal_declaration_of_human_rights.pdf	UN Universal Declaration of Human Rights
RAG with DeepSeek R1.pdf	Technical overview of the RAG architecture used

10.  Example Questions to Try

•	"What are the provisions for bail under the BNSS?"
•	"What rights does Article 12 of the UDHR guarantee?"
•	"What is the punishment for theft under the new criminal code?"
•	"Does the UDHR protect freedom of expression?"
•	"What are the rights of an accused person during trial?"

11.  Project Structure

RAG-LAW/
├── app.py                           # Streamlit UI (single entry point)
├── rag_pipeline.py                  # Backend: embeddings, FAISS, Groq LLM
├── requirements.txt                 # Pinned dependencies
├── runtime.txt                      # Python 3.11 (for Streamlit Cloud)
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml                  # Dark UI theme
│   └── secrets.toml.example         # Template — shows required secrets
└── pdfs/                            # Drop your own PDFs here
    ├── Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf
    └── universal_declaration_of_human_rights.pdf

12.  Embeddings Choice — Tradeoff Summary

Option	Chosen	Cost	Quality	API Key?
all-MiniLM-L6-v2 (HuggingFace)	✅ Yes	Free	Good	No
OpenAI text-embedding-3-small	No	Pay-per-token	Better	Yes
Cohere embed-english-v3.0	No	Free tier	Better	Yes

HuggingFace was chosen for simplicity: zero key management for embeddings, works completely inside the container, and is more than adequate for demo-scale legal document retrieval.

