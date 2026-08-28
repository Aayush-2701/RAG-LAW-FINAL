"""
rag_pipeline.py — Backend RAG module for RAG-LAW.

Responsibilities:
  - Load and cache the HuggingFace embedding model (all-MiniLM-L6-v2).
  - Build and cache a FAISS vector index from committed PDFs.
  - Build a fresh FAISS index from user-supplied PDF bytes (per-upload).
  - Retrieve relevant document chunks via similarity search.
  - Generate an answer using ChatGroq (deepseek-r1-distill-llama-70b).

No Streamlit UI code lives here — only pure backend logic imported by app.py.
"""

from __future__ import annotations

import glob
import io
import os
from pathlib import Path
from typing import Optional

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "deepseek-r1-distill-llama-70b"
PDFS_DIR = Path(__file__).parent / "pdfs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

SYSTEM_PROMPT = """You are an expert AI legal assistant. Use ONLY the pieces of \
information provided in the Context below to answer the user's question.

Rules:
- If the answer is not contained in the context, say "I don't have enough \
information in the provided documents to answer that question." Do NOT \
make up facts.
- Be concise and direct.
- Cite the source document when relevant.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Groq API key helper
# ---------------------------------------------------------------------------

def get_groq_api_key() -> Optional[str]:
    """Return the Groq API key from st.secrets or environment, or None."""
    try:
        key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    except Exception:
        key = os.environ.get("GROQ_API_KEY")
    return key or None


# ---------------------------------------------------------------------------
# Cached embedding model (loaded once per container lifetime)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_embedding_model() -> HuggingFaceEmbeddings:
    """Load and cache the sentence-transformers embedding model.

    Uses all-MiniLM-L6-v2 — ~23 MB, CPU-only, no API key required.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def _split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def _load_pdf_from_path(path: str):
    """Load a PDF from a file-system path."""
    loader = PDFPlumberLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", Path(path).name)
    return docs


def _load_pdf_from_bytes(pdf_bytes: bytes, filename: str):
    """Load a PDF from raw bytes by writing it to a temp buffer."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PDFPlumberLoader(tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
    finally:
        os.unlink(tmp_path)

    return docs


# ---------------------------------------------------------------------------
# Cached FAISS index for committed PDFs
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_preloaded_index() -> Optional[FAISS]:
    """Build (and cache) a FAISS index from all PDFs in the `pdfs/` directory.

    Also scans the repo root for any PDF that isn't already in pdfs/.
    Returns None if no PDFs are found.
    """
    # Gather PDFs: repo root + pdfs/ sub-directory
    root = Path(__file__).parent
    pdf_paths = list(root.glob("*.pdf")) + list(PDFS_DIR.glob("*.pdf"))
    # Deduplicate by resolved path
    seen, unique_paths = set(), []
    for p in pdf_paths:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            unique_paths.append(str(p))

    if not unique_paths:
        return None

    all_docs = []
    for path in unique_paths:
        try:
            all_docs.extend(_load_pdf_from_path(path))
        except Exception as exc:
            st.warning(f"Could not load {Path(path).name}: {exc}")

    if not all_docs:
        return None

    chunks = _split_documents(all_docs)
    embeddings = load_embedding_model()
    return FAISS.from_documents(chunks, embeddings)


# ---------------------------------------------------------------------------
# On-the-fly FAISS index for user-uploaded PDFs
# ---------------------------------------------------------------------------

def build_index_from_upload(pdf_bytes: bytes, filename: str) -> FAISS:
    """Build a FAISS index from a user-uploaded PDF (not cached — per upload)."""
    docs = _load_pdf_from_bytes(pdf_bytes, filename)
    if not docs:
        raise ValueError(f"Could not extract text from '{filename}'.")
    chunks = _split_documents(docs)
    embeddings = load_embedding_model()
    return FAISS.from_documents(chunks, embeddings)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_docs(query: str, vectorstore: FAISS, k: int = TOP_K):
    """Return the top-k most relevant document chunks for a query."""
    return vectorstore.similarity_search(query, k=k)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def answer_query(question: str, vectorstore: FAISS) -> tuple[str, list]:
    """Run the full RAG chain: retrieve → prompt → LLM → return (answer, docs).

    Returns:
        answer  — the LLM's string response.
        sources — the retrieved Document objects for citation display.

    Raises:
        ValueError if the Groq API key is missing.
        RuntimeError on LLM call failure.
    """
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your Streamlit secrets or "
            "set the environment variable before running."
        )

    sources = retrieve_docs(question, vectorstore)
    context = "\n\n".join(doc.page_content for doc in sources)

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    llm = ChatGroq(model=LLM_MODEL_NAME, api_key=api_key)

    chain = prompt | llm
    try:
        response = chain.invoke({"context": context, "question": question})
        answer = response.content
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    return answer, sources
